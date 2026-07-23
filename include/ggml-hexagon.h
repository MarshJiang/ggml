#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

// Returns the Hexagon architecture selected by the runtime, for example 73,
// 75, 79, or 81. Returns 0 when the Hexagon runtime cannot be initialized.
GGML_BACKEND_API int ggml_backend_hexagon_get_arch(void);

#ifdef  __cplusplus
}
#endif
