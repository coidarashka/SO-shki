#include "llama.h"
#include "common.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "nlohmann/json.hpp"
#include <android/log.h>
#include <string>
#include <vector>
#include <cstring>
#include <ctime>
#include <csignal>

// Подключаем STB Image для чтения картинок из файла в память пикселей
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

using json = nlohmann::json;

#define TAG "Mandre"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct GlobalConfig {
    int n_threads = 4;
    int n_threads_batch = 8;
    int n_ctx = 0;           // 0 = авто
    int n_batch = 512;
    int max_tokens = 2048;      
    int max_think_tokens = 512; 
    bool kv_quant = true;
    bool flash_attn = true;
};

static GlobalConfig g_conf;
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static mtmd_context* g_mtmd_ctx = nullptr;
static bool g_cancel_flag = false;

extern "C" {
    // 1. Освобождение ресурсов
    void free_engine() {
        LOGD("Mandre: [free_engine] Запуск очистки ресурсов...");
        if (g_mtmd_ctx) { LOGD("Mandre: [free] Vision"); mtmd_free(g_mtmd_ctx); g_mtmd_ctx = nullptr; }
        if (g_sampler)  { LOGD("Mandre: [free] Sampler"); llama_sampler_free(g_sampler); g_sampler = nullptr; }
        if (g_ctx)      { LOGD("Mandre: [free] Context"); llama_free(g_ctx); g_ctx = nullptr; }
        if (g_model)    { LOGD("Mandre: [free] Model"); llama_model_free(g_model); g_model = nullptr; }
        LOGD("Mandre:[free_engine] Освобождение бекенда...");
        llama_backend_free();
        LOGD("Mandre: [free_engine] Готово");
    }

    // 2. Загрузка Vision (MMPROJ / CLIP)
    int load_mmproj(const char* p) {
        LOGD("Mandre: [load_mmproj] Старт. Путь: %s", p ? p : "null");
        if (!g_model) {
            LOGE("Mandre: [load_mmproj] Ошибка: Сначала загрузите LLM!");
            return -2;
        }
        if (!p || strlen(p) == 0 || strcmp(p, "none") == 0) {
            LOGD("Mandre: [load_mmproj] Vision пропущен (нет пути).");
            return 0;
        }

        if (g_mtmd_ctx) { mtmd_free(g_mtmd_ctx); g_mtmd_ctx = nullptr; }
        
        mtmd_context_params params = mtmd_context_params_default();
        params.use_gpu = false; 
        params.image_max_tokens = 2048; // Позволяем MTMD брать до 2к токенов под качественные картинки
        
        LOGD("Mandre: [load_mmproj] Чтение MTMD CLIP файла...");
        g_mtmd_ctx = mtmd_init_from_file(p, g_model, params);
        if (g_mtmd_ctx) {
            LOGD("Mandre: [load_mmproj] Vision успешно загружен!");
            return 0;
        } else {
            LOGE("Mandre:[load_mmproj] КРИТИЧЕСКАЯ ОШИБКА: mtmd_init_from_file вернул null.");
            return -1;
        }
    }

    // 3. Конфигурация
    int configure_engine(const char* json_str) {
        LOGD("Mandre: [config] Получен JSON: %s", json_str ? json_str : "null");
        try {
            auto j = json::parse(json_str);
            if (j.contains("n_threads"))        g_conf.n_threads = j["n_threads"];
            if (j.contains("n_threads_batch"))  g_conf.n_threads_batch = j["n_threads_batch"];
            if (j.contains("n_ctx"))            g_conf.n_ctx = j["n_ctx"];
            if (j.contains("max_tokens"))       g_conf.max_tokens = j["max_tokens"];
            if (j.contains("max_think_tokens")) g_conf.max_think_tokens = j["max_think_tokens"];
            if (j.contains("kv_quant"))         g_conf.kv_quant = j["kv_quant"];
            if (j.contains("flash_attn"))       g_conf.flash_attn = j["flash_attn"];
            return 0;
        } catch (...) { return -1; }
    }

    void cancel_inference() { g_cancel_flag = true; }

    // 4. Загрузка основной модели
    int load_model(const char* p) {
        LOGD("Mandre: [load_model] Старт. Путь: %s", p);
        free_engine(); 
        llama_backend_init();

        llama_model_params mp = llama_model_default_params();
        mp.use_mmap = true;
        
        g_model = llama_model_load_from_file(p, mp);
        if (!g_model) return -1;
        g_vocab = llama_model_get_vocab(g_model);

        int final_ctx = (g_conf.n_ctx > 0) ? g_conf.n_ctx : std::min(llama_model_n_ctx_train(g_model), 4096);
        LOGD("Mandre: [load_model] Final Ctx: %d", final_ctx);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = final_ctx;
        cp.n_threads       = g_conf.n_threads;
        cp.n_threads_batch = g_conf.n_threads_batch;
        cp.n_batch         = g_conf.n_batch;
        cp.flash_attn_type = g_conf.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

        if (g_conf.kv_quant) {
            cp.type_k = GGML_TYPE_Q8_0;
            cp.type_v = GGML_TYPE_Q8_0;
        }

        g_ctx = llama_init_from_model(g_model, cp);
        if (!g_ctx) return -2;

        auto sp = llama_sampler_chain_default_params();
        g_sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist((uint32_t)time(NULL)));

        LOGD("Mandre: [load_model] LLM готова.");
        return 0;
    }

    // 5. Инференс (MTMD Vision + Текст)
    typedef void (*cb_t)(const char*);
    int infer(const char* pr, const char* img, cb_t cb) {
        LOGD("Mandre: [infer] === НОВЫЙ ЗАПРОС ===");
        if (!g_ctx || !g_sampler) { LOGE("Mandre: Контекст не инициализирован!"); return -1; }
        
        g_cancel_flag = false;
        llama_sampler_reset(g_sampler);
        
        // НОВАЯ ОЧИСТКА ПАМЯТИ
        LOGD("Mandre: [infer] Очистка KV памяти llama_memory_clear...");
        llama_memory_clear(llama_get_memory(g_ctx), true); 
        int n_past = 0;

        std::string prompt_str = pr;
        mtmd_bitmap* bmp = nullptr;

        // --- УСИЛЕННЫЙ ДЕБАГ VISION ---
        if (img && strlen(img) > 0 && strcmp(img, "none") != 0 && g_mtmd_ctx) {
            LOGD("Mandre: [Vision] Обнаружена картинка: %s", img);
            int w, h, c;
            LOGD("Mandre: [Vision] Запуск STB_IMAGE (stbi_load)...");
            unsigned char* data = stbi_load(img, &w, &h, &c, 3); // Принудительно 3 канала (RGB)
            
            if (data) {
                LOGD("Mandre: [Vision] Пиксели прочитаны! Разрешение: %dx%d, Каналов: %d", w, h, c);
                bmp = mtmd_bitmap_init(w, h, data);
                stbi_image_free(data);
                LOGD("Mandre: [Vision] mtmd_bitmap успешно создан в памяти.");

                // MTMD ожидает маркер <__media__> в промпте для замены на эмбеддинг картинки.
                if (prompt_str.find("<__media__>") == std::string::npos) {
                    LOGD("Mandre: [Vision] Маркер <__media__> не найден. Вставляем картинку в начало промпта.");
                    prompt_str = "<__media__>\n" + prompt_str;
                }
            } else {
                LOGE("Mandre: [Vision] КРИТИЧЕСКАЯ ОШИБКА STB! Невозможно прочитать файл: %s", img);
                LOGE("Mandre: [Vision] Файл может быть битым, или путь передан не в формате файловой системы.");
            }
        }

        llama_batch batch_tok = llama_batch_init(g_conf.n_batch, 0, 1);

        // Универсальный обработчик текстовых токенов
        auto process_text_tokens = [&](const llama_token* toks, size_t n_t) {
            for (size_t k = 0; k < n_t; k++) {
                batch_tok.token[batch_tok.n_tokens] = toks[k];
                batch_tok.pos[batch_tok.n_tokens] = n_past++;
                batch_tok.n_seq_id[batch_tok.n_tokens] = 1;
                batch_tok.seq_id[batch_tok.n_tokens][0] = 0;
                batch_tok.logits[batch_tok.n_tokens] = false; // Потом исправим для последнего
                batch_tok.n_tokens++;
                
                if (batch_tok.n_tokens == g_conf.n_batch) {
                    LOGD("Mandre: [infer] Отправка куска текста (%d токенов) в LLM...", batch_tok.n_tokens);
                    if (llama_decode(g_ctx, batch_tok) != 0) LOGE("Mandre: [infer] ОШИБКА llama_decode текста!");
                    batch_tok.n_tokens = 0; // Правильная очистка в новых версиях
                }
            }
        };

        if (g_mtmd_ctx) {
            LOGD("Mandre: [Vision] Запуск mtmd_tokenize...");
            mtmd_input_text input_txt;
            input_txt.text = prompt_str.c_str();
            input_txt.add_special = true;
            input_txt.parse_special = true;

            mtmd_input_chunks* chunks = mtmd_input_chunks_init();
            const mtmd_bitmap* bitmaps_arr[] = { bmp };
            int num_bitmaps = (bmp != nullptr) ? 1 : 0;
            
            int tok_res = mtmd_tokenize(g_mtmd_ctx, chunks, &input_txt, bitmaps_arr, num_bitmaps);
            if (tok_res == 0) {
                size_t n_chunks = mtmd_input_chunks_size(chunks);
                LOGD("Mandre: [Vision] mtmd_tokenize успешен! Промпт разбит на %zu блоков(chunks).", n_chunks);
                
                for (size_t i = 0; i < n_chunks; i++) {
                    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
                    int type = mtmd_input_chunk_get_type(chunk);
                    
                    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        size_t n_t = 0;
                        const llama_token* toks = mtmd_input_chunk_get_tokens_text(chunk, &n_t);
                        LOGD("Mandre: [Vision] Чанк %zu: ТЕКСТ (%zu токенов)", i, n_t);
                        process_text_tokens(toks, n_t);
                        
                    } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                        LOGD("Mandre: [Vision] Чанк %zu: ИЗОБРАЖЕНИЕ. Начат CLIP Encoding (mtmd_encode_chunk)...", i);
                        if (mtmd_encode_chunk(g_mtmd_ctx, chunk) == 0) {
                            float* embd = mtmd_get_output_embd(g_mtmd_ctx);
                            size_t n_img_toks = mtmd_input_chunk_get_n_tokens(chunk);
                            LOGD("Mandre: [Vision] Изображение закодировано. Занято токенов: %zu", n_img_toks);
                            
                            // Создаем отдельный батч специально для Эмбеддингов (указываем размер эмбеддинга)
                            int embd_dim = llama_model_n_embd_inp(g_model);
                            llama_batch batch_emb = llama_batch_init(g_conf.n_batch, embd_dim, 1);
                            
                            for (size_t k = 0; k < n_img_toks; k++) {
                                batch_emb.pos[batch_emb.n_tokens] = n_past++;
                                batch_emb.n_seq_id[batch_emb.n_tokens] = 1;
                                batch_emb.seq_id[batch_emb.n_tokens][0] = 0;
                                batch_emb.logits[batch_emb.n_tokens] = false;
                                
                                memcpy(batch_emb.embd + batch_emb.n_tokens * embd_dim,
                                       embd + k * embd_dim,
                                       embd_dim * sizeof(float));
                                       
                                batch_emb.n_tokens++;
                                if (batch_emb.n_tokens == g_conf.n_batch) {
                                    LOGD("Mandre: [Vision] Отправка части картинки (батч %d) в LLM...", batch_emb.n_tokens);
                                    if(llama_decode(g_ctx, batch_emb) != 0) LOGE("Mandre: [Vision] ОШИБКА llama_decode картинки!");
                                    batch_emb.n_tokens = 0;
                                }
                            }
                            if (batch_emb.n_tokens > 0) {
                                LOGD("Mandre: [Vision] Отправка остатка картинки (%d токенов) в LLM...", batch_emb.n_tokens);
                                if(llama_decode(g_ctx, batch_emb) != 0) LOGE("Mandre: [Vision] ОШИБКА llama_decode картинки!");
                            }
                            llama_batch_free(batch_emb);
                        } else {
                            LOGE("Mandre: [Vision] КРИТИЧЕСКАЯ ОШИБКА: mtmd_encode_chunk провалился!");
                        }
                    }
                }
            } else {
                LOGE("Mandre: [Vision] ОШИБКА mtmd_tokenize. Код: %d", tok_res);
            }
            
            mtmd_input_chunks_free(chunks);
            if (bmp) mtmd_bitmap_free(bmp);
            
        } else {
            // Обычный текстовый инференс без Vision
            LOGD("Mandre: [infer] Vision не загружен. Обычная токенизация...");
            std::vector<llama_token> tk(prompt_str.size() + 16);
            int n = llama_tokenize(g_vocab, prompt_str.c_str(), prompt_str.size(), tk.data(), tk.size(), true, true);
            tk.resize(n);
            LOGD("Mandre: [infer] Промпт разбит на %d токенов", n);
            process_text_tokens(tk.data(), n);
        }
        
        // Помечаем самый последний токен как требующий logits
        if (batch_tok.n_tokens > 0) {
            batch_tok.logits[batch_tok.n_tokens - 1] = true;
            LOGD("Mandre: [infer] Декодирование финального текстового батча (%d токенов)...", batch_tok.n_tokens);
            if (llama_decode(g_ctx, batch_tok) != 0) LOGE("Mandre: [infer] ОШИБКА llama_decode!");
            batch_tok.n_tokens = 0;
        }

        LOGD("Mandre: [infer] Контекст заполнен (n_past = %d). Старт генерации!", n_past);
        int think_tokens = 0;
        bool in_think_tag = false;

        for (int i = 0; i < g_conf.max_tokens; i++) {
            if (g_cancel_flag) break;

            llama_token id = llama_sampler_sample(g_sampler, g_ctx, -1);
            if (llama_vocab_is_eog(g_vocab, id)) break;

            char b[256];
            int n_p = llama_token_to_piece(g_vocab, id, b, sizeof(b), 0, true);
            if (n_p > 0) {
                std::string piece(b, n_p);

                if (piece.find("<think>") != std::string::npos) in_think_tag = true;
                
                if (in_think_tag) {
                    think_tokens++;
                    if (think_tokens > g_conf.max_think_tokens) {
                        cb("\n[Think Limit Exceeded]\n");
                        in_think_tag = false;
                    }
                }
                cb(piece.c_str());
                if (piece.find("</think>") != std::string::npos) in_think_tag = false;
            }

            llama_sampler_accept(g_sampler, id);
            
            // Отправляем сгенерированный токен обратно в контекст
            batch_tok.n_tokens = 0; // Сброс
            batch_tok.token[0] = id;
            batch_tok.pos[0] = n_past++;
            batch_tok.n_seq_id[0] = 1;
            batch_tok.seq_id[0][0] = 0;
            batch_tok.logits[0] = true;
            batch_tok.n_tokens = 1;

            if (llama_decode(g_ctx, batch_tok) != 0) {
                LOGE("Mandre: [infer] ОШИБКА llama_decode при генерации токена %d!", i);
                break;
            }
        }
        
        LOGD("Mandre: [infer] Инференс завершен!");
        llama_batch_free(batch_tok);
        return 0;
    }
}
