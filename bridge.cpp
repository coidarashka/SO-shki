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

using json = nlohmann::json;

#define TAG "MandreAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct GlobalConfig {
    int n_threads = 4;
    int n_threads_batch = 8; // Больше потоков для промпта
    int n_ctx = 0;           // 0 = авто из модели
    int n_batch = 512;
    int max_tokens = 2048;
    int max_think_tokens = 512; // Отдельный лимит на размышления
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
    int configure_engine(const char* json_str) {
        try {
            auto j = json::parse(json_str);
            if (j.contains("n_threads"))      g_conf.n_threads = j["n_threads"];
            if (j.contains("n_threads_batch")) g_conf.n_threads_batch = j["n_threads_batch"];
            if (j.contains("n_ctx"))          g_conf.n_ctx = j["n_ctx"];
            if (j.contains("max_tokens"))     g_conf.max_tokens = j["max_tokens"];
            if (j.contains("max_think_tokens")) g_conf.max_think_tokens = j["max_think_tokens"];
            if (j.contains("kv_quant"))       g_conf.kv_quant = j["kv_quant"];
            return 0;
        } catch (...) { return -1; }
    }

    void cancel_inference() { g_cancel_flag = true; }

    // Чистка ресурсов перед новой загрузкой для предотвращения вылетов
    void free_engine() {
        if (g_mtmd_ctx) { mtmd_free(g_mtmd_ctx); g_mtmd_ctx = nullptr; }
        if (g_sampler)  { llama_sampler_free(g_sampler); g_sampler = nullptr; }
        if (g_ctx)      { llama_free(g_ctx); g_ctx = nullptr; }
        if (g_model)    { llama_model_free(g_model); g_model = nullptr; }
        llama_backend_free();
    }

    int load_model(const char* p) {
        free_engine(); // Важно: очищаем всё старое
        llama_backend_init();
        
        llama_model_params mp = llama_model_default_params();
        mp.use_mmap = true;
        g_model = llama_model_load_from_file(p, mp);
        if (!g_model) return -1;
        
        g_vocab = llama_model_get_vocab(g_model);

        // Адаптивный контекст: если юзер не задал, берем из модели
        int model_ctx = llama_model_n_ctx_train(g_model);
        int final_ctx = (g_conf.n_ctx > 0) ? g_conf.n_ctx : std::min(model_ctx, 4096);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = final_ctx;
        cp.n_threads       = g_conf.n_threads;
        cp.n_threads_batch = g_conf.n_threads_batch;
        cp.flash_attn      = g_conf.flash_attn; // Оптимизация памяти/скорости

        if (g_conf.kv_quant) {
            cp.type_k = GGML_TYPE_Q8_0;
            cp.type_v = GGML_TYPE_Q8_0;
        }

        g_ctx = llama_init_from_model(g_model, cp);
        if (!g_ctx) return -2;

        auto sp = llama_sampler_chain_default_params();
        g_sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist((uint32_t)time(NULL)));
        
        LOGD("Model Loaded. Adaptive Context: %d", final_ctx);
        return 0;
    }

    typedef void (*cb_t)(const char*);
    
    int infer(const char* pr, const char* img, cb_t cb) {
        if (!g_ctx || !g_sampler) return -1;
        g_cancel_flag = false;
        
        // Сброс состояния самплера для предотвращения "зацикливания" на 2-й генерации
        llama_sampler_reset(g_sampler);
        llama_memory_seq_rm(llama_get_memory(g_ctx), -1, -1, -1);

        // Токенизация и ввод...
        std::vector<llama_token> tk(strlen(pr) + 16);
        int n = llama_tokenize(g_vocab, pr, strlen(pr), tk.data(), tk.size(), true, true);
        tk.resize(n);
        llama_decode(g_ctx, llama_batch_get_one(tk.data(), n));

        int think_tokens_count = 0;
        int answer_tokens_count = 0;
        bool is_thinking = false;

        for (int i = 0; i < g_conf.max_tokens; i++) {
            if (g_cancel_flag) break;

            llama_token id = llama_sampler_sample(g_sampler, g_ctx, -1);
            if (llama_vocab_is_eog(g_vocab, id)) break;

            char buf[256];
            int n_piece = llama_token_to_piece(g_vocab, id, buf, sizeof(buf), 0, true);
            if (n_piece > 0) {
                std::string piece(buf, n_piece);
                
                // Логика разделения Think/Answer
                if (piece.find("<think>") != std::string::npos) is_thinking = true;
                
                if (is_thinking) {
                    think_tokens_count++;
                    if (think_tokens_count > g_conf.max_think_tokens) {
                        // Если превысили лимит размышлений, принудительно выходим
                        cb("\n[Лимит размышлений превышен]\n");
                        break; 
                    }
                } else {
                    answer_tokens_count++;
                }

                if (piece.find("</think>") != std::string::npos) is_thinking = false;

                cb(piece.c_str());
            }

            llama_sampler_accept(g_sampler, id);
            if (llama_decode(g_ctx, llama_batch_get_one(&id, 1)) != 0) break;
        }
        return 0;
    }
}
