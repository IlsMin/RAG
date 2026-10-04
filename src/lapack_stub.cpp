// FAISS вызывает QR-разложение только из PCA/VectorTransform.
// Плоский индекс IndexFlatIP эти функции не использует, а в OpenBLAS их нет.

extern "C" {

int sgeqrf_(int * /*m*/,
            int * /*n*/,
            float * /*a*/,
            int * /*lda*/,
            float * /*tau*/,
            float * /*work*/,
            int * /*lwork*/,
            int *info)
{
    if (info)
        *info = 1;
    return 0;
}

int sorgqr_(int * /*m*/,
            int * /*n*/,
            int * /*k*/,
            float * /*a*/,
            int * /*lda*/,
            float * /*tau*/,
            float * /*work*/,
            int * /*lwork*/,
            int *info)
{
    if (info)
        *info = 1;
    return 0;
}

} // extern "C"
