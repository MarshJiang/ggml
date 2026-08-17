#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>
#include <math.h>
#include <stdint.h>

#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"

struct htp_snake_context {
    struct htp_ops_context * octx;
    uint32_t                  rows_per_thread;
    uint32_t                  rows;
};

static void snake_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_snake_context * sctx = (const struct htp_snake_context *) data;
    const struct htp_ops_context   * octx = sctx->octx;
    const struct htp_tensor        * in0  = octx->src[0];
    const struct htp_tensor        * in1  = octx->src[1];
    const struct htp_tensor        * x    = in0->ne[0] == octx->dst->ne[0] && in0->ne[1] == octx->dst->ne[1] ? in0 : in1;
    const struct htp_tensor        * a    = x == in0 ? in1 : in0;
    const struct htp_tensor        * invb = octx->src[2];
    const struct htp_tensor        * dst  = octx->dst;

    const uint32_t begin = MIN(sctx->rows_per_thread * ith, sctx->rows);
    const uint32_t end   = MIN(begin + sctx->rows_per_thread, sctx->rows);
    const uint32_t width = x->ne[0];
    const uint32_t nvec  = width / VLEN_FP32;
    const uint32_t tail  = width % VLEN_FP32;

    for (uint32_t row = begin; row < end; ++row) {
        const float av = *(const float *) ((const uint8_t *) (uintptr_t) a->data + (size_t) row * a->nb[1]);
        const float bv = *(const float *) ((const uint8_t *) (uintptr_t) invb->data + (size_t) row * invb->nb[1]);
        const HVX_Vector va = hvx_vec_splat_f32(av);
        const HVX_Vector vb = hvx_vec_splat_f32(bv);

        const float * src = (const float *) ((const uint8_t *) (uintptr_t) x->data + (size_t) row * x->nb[1]);
        float * out       = (float *) ((uint8_t *) (uintptr_t) dst->data + (size_t) row * dst->nb[1]);

        for (uint32_t i = 0; i < nvec; ++i) {
            const HVX_Vector vx = hvx_vmemu(src + i * VLEN_FP32);
            const HVX_Vector vs = hvx_vec_sin_f32(hvx_vec_mul_f32_f32(va, vx));
            const HVX_Vector vy = hvx_vec_add_f32_f32(vx,
                hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(vs, vs), vb));
            hvx_vmemu(out + i * VLEN_FP32) = vy;
        }

        const uint32_t offset = nvec * VLEN_FP32;
        for (uint32_t i = 0; i < tail; ++i) {
            const float xv = src[offset + i];
            const float sv = sinf(av * xv);
            out[offset + i] = xv + sv * sv * bv;
        }
    }

    (void) nth;
}

int op_snake(struct htp_ops_context * octx) {
    const struct htp_tensor * in0  = octx->src[0];
    const struct htp_tensor * in1  = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_tensor * x    = in0 && dst && in0->ne[0] == dst->ne[0] && in0->ne[1] == dst->ne[1] ? in0 : in1;
    const struct htp_tensor * a    = x == in0 ? in1 : in0;
    const struct htp_tensor * invb = octx->src[2];

    if (!x || !a || !invb || x->type != HTP_TYPE_F32 || a->type != HTP_TYPE_F32 ||
        invb->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (x->ne[2] != 1 || x->ne[3] != 1 || a->ne[0] != 1 || invb->ne[0] != 1 ||
        a->ne[1] != x->ne[1] || invb->ne[1] != x->ne[1]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t rows = x->ne[1];
    const uint32_t n_threads = MIN(rows, octx->n_threads);
    struct htp_snake_context sctx = {
        .octx            = octx,
        .rows_per_thread = (rows + n_threads - 1) / n_threads,
        .rows            = rows,
    };
    worker_pool_run_func(octx->ctx->worker_pool, snake_thread, &sctx, n_threads);
    return HTP_STATUS_OK;
}
