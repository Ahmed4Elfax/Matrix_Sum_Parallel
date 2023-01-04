#include <bits/stdc++.h>
#include <iostream>
#include <chrono>

using namespace std;


// initialize A as lower triangular matrix with ones
// initialize B as upper triangular matrix with ones
template <typename value_t,typename index_t>
void init(std::vector<value_t>& A,
          std::vector<value_t>& B,
          index_t n) {

    for (index_t row = 0; row < n; row++)
        for (index_t col = 0; col < n; col++)
            A[row*n+col] = row >= col ? 1 : 0;

    for (index_t row = 0; row < n; row++)
        for (index_t col = 0; col < n; col++)
            B[row*n+col] = row <= col ? 1 : 0;
}

// the sequential sum of 2 matrices
template <typename value_t, typename index_t>
void sequential_add(
        std::vector<value_t>& A, std::vector<value_t>& B, std::vector<value_t>& C, index_t n) {

    for (index_t row = 0; row < n; row++) {
        for (index_t col = 0; col < n; col++)
            C[row*n + col] = A[row*n + col] + B[row*n + col];
    }
}



// Matrix sum template using a static BLOCK distribution template
template <typename value_t, typename index_t>
void block_parallel_sum (std::vector<value_t>& A, std::vector<value_t>& B,
                         std::vector <value_t>& C, index_t n, index_t num_threads = 8) {
    auto block = [&] (const index_t& id) -> void {
        for (index_t row = id; row < n; row+=num_threads) {
            for (index_t col = 0; col < n; col++) {
                C[row*n + col] = A[row*n + col] + B[row*n + col];
            }
        }
    };
    std::vector<std::thread> threads;
    for (index_t id = 0; id < num_threads; id++) threads.emplace_back(block, id);
    for (auto& thread : threads) thread.join();
}


// Matrix sum template using a static BLOCK_CYCLIC distribution template
template <typename value_t, typename index_t>
void block_cyclic_parallel_sum(std::vector<value_t>& A, std::vector<value_t>& B,
                        std::vector <value_t>& C, index_t n, index_t num_threads = 8,
                        index_t chunk_size=64/sizeof(value_t)) {
    auto block = [&](const index_t& id) -> void {
        const index_t offset = id*chunk_size;
        const index_t stride = num_threads*chunk_size;
        for (index_t lower = offset; lower < n; lower += stride){
            const index_t upper = std::min(lower + chunk_size, n);
            for (index_t row = lower; row < upper; row++) {
                for (index_t col = 0; col < n; col++)
                    C[row*n + col] = A[row*n + col] + B[row*n + col];
            }
        }
    };
    std::vector<thread> threads;
    for (index_t id = 0; id < num_threads; id++) threads.emplace_back(block, id);
    for (auto& thread : threads) thread.join();
}

// Matrix sum template using a static CYCLIC distribution template
template <typename value_t, typename index_t>
void cyclic_parallel_sum(std::vector<value_t>& A, std::vector<value_t>& B,
                        std::vector <value_t>& C, index_t n, index_t num_threads = 8) {
    auto block = [&](const index_t& id) -> void {
        for (index_t row = id; row < n; row+=num_threads){
            for (index_t col = 0; col < n; col++)
                C[row*n + col] = A[row*n + col] + B[row*n + col];
        }
    };
    std::vector<thread> threads;
    for (index_t id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);
    for (auto& thread : threads) thread.join();
}

int main(int argc, char* argv[]) {
    const uint64_t n = 16;

    std::vector<uint64_t> A(n*n);   // alloc
    std::vector<uint64_t> B(n*n);
    std::vector<uint64_t> C(n*n);

    init(A, B, n);               // init
    auto start = std::chrono::high_resolution_clock::now();
    cyclic_parallel_sum(A, B, C, n); // add
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout<<duration.count()<<endl;
    cout<<"A:"<<endl;
    for (uint64_t row = 0; row < n; row++){
        for (uint64_t col = 0; col < n; col++)
            cout<<A[row*n+col]<<" ";
        cout << endl;
    }
    cout<<"B:"<<endl;

    for (uint64_t row = 0; row < n; row++){
        for (uint64_t col = 0; col < n; col++)
            cout<<B[row*n+col]<<" ";
        cout << endl;
    }

    cout<<"C:"<<endl;

    for (uint64_t row = 0; row < n; row++){
        for (uint64_t col = 0; col < n; col++)
            cout<<C[row*n+col]<<" ";
        cout << endl;
    }

}
