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

using json = nlohmann::json;

// --- ЗАДАЧА 1: Тег Mandre и полный дебаг ---
#define TAG "Mandre"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct GlobalConfig {
    int n_threads = 4;
    int n_threads_batch = 8;
    int n_ctx = 0;           // 0 = авто
    int n_batch = 512;
    int max_tokens = 2048;      // Общий лимит ответа
    int max_think_tokens = 512;  // Лимит только для <think>
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
        if (g_mtmd_ctx) { LOGD("Mandre: [free_engine] Освобождение Vision контекста"); mtmd_free(g_mtmd_ctx); g_mtmd_ctx = nullptr; }
        if (g_sampler)  { LOGD("Mandre: [free_engine] Освобождение семплера"); llama_sampler_free(g_sampler); g_sampler = nullptr; }
        if (g_ctx)      { LOGD("Mandre: [free_engine] Освобождение LLM контекста"); llama_free(g_ctx); g_ctx = nullptr; }
        if (g_model)    { LOGD("Mandre: [free_engine] Освобождение модели"); llama_model_free(g_model); g_model = nullptr; }
        LOGD("Mandre: [free_engine] Освобождение бекенда llama");
        llama_backend_free();
        LOGD("Mandre: [free_engine] Очистка успешно завершена");
    }

    // 2. Загрузка Vision (MMPROJ)
    int load_mmproj(const char* p) {
        LOGD("Mandre: [load_mmproj] Вызов загрузки Vision. Путь: %s", p ? p : "null");
        if (!g_model) {
            LOGE("Mandre: [load_mmproj] Ошибка: g_model = null. Сначала загрузите основную модель!");
            return -2;
        }
        if (!p || strlen(p) == 0 || strcmp(p, "none") == 0) {
            LOGD("Mandre: [load_mmproj] Путь к фото-модели пуст или 'none'. Пропускаем инициализацию Vision.");
            return 0;
        }

        if (g_mtmd_ctx) {
            LOGD("Mandre: [load_mmproj] Обнаружен старый Vision контекст, удаляем...");
            mtmd_free(g_mtmd_ctx);
            g_mtmd_ctx = nullptr;
        }
        
        LOGD("Mandre: [load_mmproj] Настройка параметров Vision (max_tokens=128)");
        mtmd_context_params params = mtmd_context_params_default();
        params.use_gpu = false; 
        params.image_max_tokens = 128; // Фиксированный лимит для картинок
        
        LOGD("Mandre: [load_mmproj] Чтение файла mtmd...");
        g_mtmd_ctx = mtmd_init_from_file(p, g_model, params);
        if (g_mtmd_ctx) {
            LOGD("Mandre: [load_mmproj] Vision модель успешно загружена");
            return 0;
        } else {
            LOGE("Mandre: [load_mmproj] КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить Vision модель");
            return -1;
        }
    }

    // 3. Конфигурация параметров
    int configure_engine(const char* json_str) {
        LOGD("Mandre: [configure_engine] Получен JSON: %s", json_str ? json_str : "null");
        try {
            auto j = json::parse(json_str);
            if (j.contains("n_threads"))        { g_conf.n_threads = j["n_threads"]; LOGD("Mandre: n_threads = %d", g_conf.n_threads); }
            if (j.contains("n_threads_batch"))  { g_conf.n_threads_batch = j["n_threads_batch"]; LOGD("Mandre: n_threads_batch = %d", g_conf.n_threads_batch); }
            if (j.contains("n_ctx"))            { g_conf.n_ctx = j["n_ctx"]; LOGD("Mandre: n_ctx = %d", g_conf.n_ctx); }
            if (j.contains("max_tokens"))       { g_conf.max_tokens = j["max_tokens"]; LOGD("Mandre: max_tokens = %d", g_conf.max_tokens); }
            if (j.contains("max_think_tokens")) { g_conf.max_think_tokens = j["max_think_tokens"]; LOGD("Mandre: max_think_tokens = %d", g_conf.max_think_tokens); }
            if (j.contains("kv_quant"))         { g_conf.kv_quant = j["kv_quant"]; LOGD("Mandre: kv_quant = %d", g_conf.kv_quant); }
            if (j.contains("flash_attn"))       { g_conf.flash_attn = j["flash_attn"]; LOGD("Mandre: flash_attn = %d", g_conf.flash_attn); }
            LOGD("Mandre: [configure_engine] Конфигурация применена успешно");
            return 0;
        } catch (const std::exception& e) { 
            LOGE("Mandre: [configure_engine] Ошибка парсинга JSON: %s", e.what());
            return -1; 
        }
    }

    void cancel_inference() { 
        LOGD("Mandre: [cancel_inference] Установлен флаг отмены генерации!");
        g_cancel_flag = true; 
    }

    // 4. Загрузка основной модели
    int load_model(const char* p) {
        LOGD("Mandre: [load_model] Старт загрузки основной модели. Путь: %s", p);
        free_engine(); 
        llama_backend_init();

        LOGD("Mandre: [load_model] Применение параметров llama_model_default_params");
        llama_model_params mp = llama_model_default_params();
        mp.use_mmap = true;
        
        LOGD("Mandre: [load_model] Чтение файла LLM...");
        g_model = llama_model_load_from_file(p, mp);
        if (!g_model) {
            LOGE("Mandre: [load_model] Ошибка загрузки файла модели!");
            return -1;
        }
        g_vocab = llama_model_get_vocab(g_model);
        LOGD("Mandre: [load_model] Модель загружена, vocab получен");

        // Адаптивный контекст
        int model_train_ctx = llama_model_n_ctx_train(g_model);
        int final_ctx = (g_conf.n_ctx > 0) ? g_conf.n_ctx : std::min(model_train_ctx, 4096);
        LOGD("Mandre: [load_model] Определение контекста. Train ctx: %d, Final ctx: %d", model_train_ctx, final_ctx);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = final_ctx;
        cp.n_threads       = g_conf.n_threads;
        cp.n_threads_batch = g_conf.n_threads_batch;
        cp.n_batch         = g_conf.n_batch;
        
        LOGD("Mandre: [load_model] Flash Attention = %d", g_conf.flash_attn);
        cp.flash_attn_type = g_conf.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

        if (g_conf.kv_quant) {
            LOGD("Mandre: [load_model] Включено квантование KV кеша (Q8_0)");
            cp.type_k = GGML_TYPE_Q8_0;
            cp.type_v = GGML_TYPE_Q8_0;
        }

        LOGD("Mandre: [load_model] Инициализация контекста LLM...");
        g_ctx = llama_init_from_model(g_model, cp);
        if (!g_ctx) {
            LOGE("Mandre: [load_model] Ошибка инициализации llama_context!");
            return -2;
        }

        LOGD("Mandre: [load_model] Настройка семплера...");
        auto sp = llama_sampler_chain_default_params();
        g_sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist((uint32_t)time(NULL)));

        LOGD("Mandre: [load_model] Движок полностью готов. Адаптивный контекст: %d", final_ctx);
        return 0;
    }

    // 5. Инференс
    typedef void (*cb_t)(const char*);
    int infer(const char* pr, const char* img, cb_t cb) {
        LOGD("Mandre: [infer] --- НОВЫЙ ЗАПРОС ИНФЕРЕНСА ---");
        LOGD("Mandre: [infer] Промпт (длина %zu), Изображение: %s", strlen(pr), img ? img : "отсутствует");

        if (!g_ctx || !g_sampler) {
            LOGE("Mandre: [infer] Ошибка: контекст или семплер не инициализированы!");
            return -1;
        }
        g_cancel_flag = false;

        LOGD("Mandre: [infer] Сброс семплера и полная очистка KV-кэша");
        llama_sampler_reset(g_sampler);
        
        // --- ЗАДАЧА 2: ИСПРАВЛЕНИЕ ВЫЛЕТА ПРИ 2-Й ГЕНЕРАЦИИ ---
        // Старый метод (llama_memory_seq_rm) устарел и вызывал смещения/ошибки. 
        // Используем полную жесткую очистку кэша:
        llama_kv_cache_clear(g_ctx); 

        int n_past = 0; // Позиция текущего токена в памяти контекста

        // Лямбда для безопасного добавления токена в batch
        auto batch_add = [](llama_batch& b, llama_token id, llama_pos pos, bool logits) {
            b.token[b.n_tokens] = id;
            b.pos[b.n_tokens] = pos;
            b.n_seq_id[b.n_tokens] = 1;
            b.seq_id[b.n_tokens][0] = 0;
            b.logits[b.n_tokens] = logits;
            b.n_tokens++;
        };

        // --- ЗАДАЧА 3: ИСПРАВЛЕНИЕ БАГА "ИИ НЕ ВИДИТ ФОТО" ---
        // Ранее переменная img никак не использовалась. Теперь энкодим фото:
        if (img && strlen(img) > 0 && strcmp(img, "none") != 0 && g_mtmd_ctx) {
            LOGD("Mandre: [infer] Найдена картинка. Запуск оценки изображения...");
            
            // Стандартный подход для форков llava/mtmd
            mtmd_image_embed * embed = mtmd_image_embed_make_with_filename(g_mtmd_ctx, g_conf.n_threads, img);
            if (embed) {
                LOGD("Mandre: [infer] Embedding картинки создан. Отправка в контекст...");
                bool eval_ok = mtmd_eval_image_embed(g_ctx, embed, g_conf.n_batch, &n_past);
                mtmd_image_embed_free(embed);
                
                if (!eval_ok) {
                    LOGE("Mandre: [infer] Ошибка при декодировании картинки (mtmd_eval_image_embed)");
                } else {
                    LOGD("Mandre: [infer] Картинка успешно обработана. Занято токенов (n_past): %d", n_past);
                }
            } else {
                LOGE("Mandre: [infer] Не удалось прочитать/создать embed картинки (неверный путь или битый файл?)");
            }
        } else if (img && strcmp(img, "none") != 0) {
            LOGD("Mandre: [infer] Картинка указана, но g_mtmd_ctx=null (Модель Vision не загружена!)");
        }

        LOGD("Mandre: [infer] Токенизация текстового промпта...");
        std::vector<llama_token> tk(strlen(pr) + 16);
        int n = llama_tokenize(g_vocab, pr, strlen(pr), tk.data(), tk.size(), true, true);
        tk.resize(n);
        LOGD("Mandre: [infer] Промпт разбит на %d токенов", n);

        llama_batch batch = llama_batch_init(g_conf.n_batch, 0, 1);

        LOGD("Mandre: [infer] Декодирование промпта в память (Batching)...");
        for (int i = 0; i < n; i++) {
            bool is_last = (i == n - 1);
            batch_add(batch, tk[i], n_past++, is_last);
            
            if (batch.n_tokens == g_conf.n_batch || is_last) {
                LOGD("Mandre: [infer] Отправка батча размером %d токенов...", batch.n_tokens);
                if (llama_decode(g_ctx, batch) != 0) {
                    LOGE("Mandre: [infer] ОШИБКА llama_decode при обработке промпта!");
                    llama_batch_free(batch);
                    return -1;
                }
                llama_batch_clear(&batch);
            }
        }

        int think_tokens = 0;
        bool in_think_tag = false;

        LOGD("Mandre: [infer] --- СТАРТ ЦИКЛА ГЕНЕРАЦИИ ОТВЕТА ---");
        for (int i = 0; i < g_conf.max_tokens; i++) {
            if (g_cancel_flag) {
                LOGD("Mandre: [infer] Генерация прервана пользователем (cancel_flag)");
                break;
            }

            if (n_past >= llama_n_ctx(g_ctx)) {
                LOGE("Mandre: [infer] Достигнут лимит контекста (%d). Остановка.", n_past);
                break;
            }

            llama_token id = llama_sampler_sample(g_sampler, g_ctx, -1);
            
            if (llama_vocab_is_eog(g_vocab, id)) {
                LOGD("Mandre: [infer] Встречен токен окончания (EOG). Завершение.");
                break;
            }

            char b[256];
            int n_p = llama_token_to_piece(g_vocab, id, b, sizeof(b), 0, true);
            if (n_p > 0) {
                std::string piece(b, n_p);

                if (piece.find("<think>") != std::string::npos) {
                    LOGD("Mandre: [infer] Обнаружен тег <think>");
                    in_think_tag = true;
                }
                
                if (in_think_tag) {
                    think_tokens++;
                    if (think_tokens > g_conf.max_think_tokens) {
                        LOGD("Mandre: [infer] Превышен лимит размышлений (think_tokens > %d)", g_conf.max_think_tokens);
                        cb("\n[Think Limit Exceeded]\n");
                        in_think_tag = false;
                    }
                }

                cb(piece.c_str());
                
                if (piece.find("</think>") != std::string::npos) {
                    LOGD("Mandre: [infer] Обнаружен тег </think>");
                    in_think_tag = false;
                }
            }

            llama_sampler_accept(g_sampler, id);
            
            llama_batch_clear(&batch);
            batch_add(batch, id, n_past++, true);

            if (llama_decode(g_ctx, batch) != 0) {
                LOGE("Mandre: [infer] ОШИБКА llama_decode при генерации токена %d!", i);
                break;
            }
        }
        
        LOGD("Mandre: [infer] Инференс завершен. Всего токенов в контексте: %d", n_past);
        llama_batch_free(batch);
        return 0;
    }
}
