#include <bits/stdc++.h>
#include <iostream>
#include <chrono>
#include <mutex> // std::mutex, std::lock_guard


using namespace std;
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

template <typename value_t, typename index_t>
void printInitials(vector<value_t>& A, vector<value_t>& B,index_t n){
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
}

// initialize A as lower triangular matrix with ones
// initialize B as upper triangular matrix with ones
template <typename value_t,typename index_t>
void init(vector<value_t>& A,vector<value_t>& B,index_t n) {
    for (index_t row = 0; row < n; row++)
        for (index_t col = 0; col < n; col++)
            A[row*n+col] = row >= col ? 1 : 0;

    for (index_t row = 0; row < n; row++)
        for (index_t col = 0; col < n; col++)
            B[row*n+col] = row <= col ? 1 : 0;

//    printInitials(A,B,n);
}

// the sequential sum of 2 matrices
template <typename value_t, typename index_t>
long long int sequential_add(vector<value_t>& A, vector<value_t>& B, vector<value_t>& C, index_t n) {
    auto start = chrono::high_resolution_clock::now();
    for (index_t row = 0; row < n; row++) {
        for (index_t col = 0; col < n; col++)
            C[row*n + col] = A[row*n + col] + B[row*n + col];
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    return duration.count();
}



// Matrix sum template using a static BLOCK distribution template
template <typename value_t, typename index_t>
long long int block_parallel_sum (vector<value_t>& A, vector<value_t>& B,vector <value_t>& C, index_t n, index_t num_threads = 8) {
    auto start = chrono::high_resolution_clock::now();
    auto block = [&](const index_t& id) -> void {
        const index_t chunk = ceil((double)n/num_threads);
        const index_t lower = id*chunk;
        const index_t upper = std::min(lower + chunk, n);
        for (index_t row = lower; row < upper; row++) {
            for (index_t col = 0; col < n; col++)
                C[row*n + col] = A[row*n + col] + B[row*n + col];
        }
    };
    vector<thread> threads;
    for (index_t id = 0; id < num_threads; id++) threads.emplace_back(block, id);
    for (auto& thread : threads) thread.join();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    return duration.count();
}


// Matrix sum template using a static BLOCK_CYCLIC distribution template
template <typename value_t, typename index_t>
long long int block_cyclic_parallel_sum(vector<value_t>& A, vector<value_t>& B,vector <value_t>& C, index_t n, index_t num_threads = 8,
                        index_t chunk_size=64/sizeof(value_t)) {
    auto start = chrono::high_resolution_clock::now();
    auto block = [&](const index_t& id) -> void {
        const index_t offset = id*chunk_size;
        const index_t stride = num_threads*chunk_size;
        for (index_t lower = offset; lower < n; lower += stride){
            const index_t upper = min(lower + chunk_size, n);
            for (index_t row = lower; row < upper; row++) {
                for (index_t col = 0; col < n; col++)
                    C[row*n + col] = A[row*n + col] + B[row*n + col];
            }
        }
    };
    vector<thread> threads;
    for (index_t id = 0; id < num_threads; id++) threads.emplace_back(block, id);
    for (auto& thread : threads) thread.join();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    return duration.count();
}

// Matrix sum template using a static CYCLIC distribution template
template <typename value_t, typename index_t>
long long int cyclic_parallel_sum(vector<value_t>& A, vector<value_t>& B,vector <value_t>& C, index_t n, index_t num_threads = 8) {
    auto start = chrono::high_resolution_clock::now();
    auto block = [&](const index_t& id) -> void {
        for (index_t row = id; row < n; row+=num_threads){
            for (index_t col = 0; col < n; col++)
                C[row*n + col] = A[row*n + col] + B[row*n + col];
        }
    };
    vector<thread> threads;
    for (index_t id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);
    for (auto& thread : threads) thread.join();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    return duration.count();
}

// Matrix sum template using a Dynamic Block distribution template
template <typename index_t, typename value_t>
long long int sum_dynamic(vector<value_t>& A, vector<value_t>& B,vector <value_t>& C, index_t n,index_t num_threads = 8, index_t chunk_size = 64 / sizeof(value_t)) {
    auto start = chrono::high_resolution_clock::now();
    std::mutex mutex; // declare mutex and current lower index
    index_t global_lower = 0;
    auto dynamic_block_cyclic = [&]() -> void {
        index_t lower = 0;         // assume we have not done anything
        while (lower < n) {     // while there are still rows to compute
            {                      // update lower row with global lower row
                std::lock_guard<std::mutex> lock_guard(mutex);
                lower = global_lower;
                global_lower += chunk_size;
            } // here we release the lock
            const index_t upper = std::min(lower+chunk_size, n); // compute upper border of block
            for (index_t row = lower; row < upper; row++) {
                for (index_t col = 0; col < n; col++)
                    C[row*n + col] = A[row*n + col] + B[row*n + col];
            }
        }
    };
    std::vector<std::thread> threads;
    for (index_t id = 0; id < num_threads; id++) threads.emplace_back(dynamic_block_cyclic);
    for (auto& thread : threads) thread.join();
    return chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
}



template <typename value_t, typename index_t>
void printResults(vector <value_t>& C,index_t n, const string& title){
    cout<< title<<":"<<endl;
    for (uint64_t row = 0; row < n; row++){
        for (uint64_t col = 0; col < n; col++)
            cout<<C[row*n+col]<<" ";
        cout << endl;
    }
}

int main() {
    std::ofstream results;
    results.open ("results.csv");
    results << ",,block_parallel_sum,,,,,,,block_cyclic_parallel_sum,,,,,,,cyclic_parallel_sum\n";
    results << "Matrix Size,sequential add" << endl;
    uint64_t size = 1;
    for (int i = 1 ; i < 15 ; i++){
        size <<= 1;
//        cout << size << endl;
        vector<uint64_t> A(size*size);   // alloc
        vector<uint64_t> B(size*size);
        vector<uint64_t> C(size*size);
        init(A, B, size); // init
        results << size << "," << sequential_add(A,B,C,size);
        for (uint64_t threads = 8 ; threads < 513; threads*=2){
//            cout << threads << endl;
            results <<","<<block_parallel_sum(A,B,C,size,threads);
        }
        for (uint64_t threads = 8 ; threads < 513; threads*=2){
            results <<","<<block_cyclic_parallel_sum(A,B,C,size,threads);
        }
        for (uint64_t threads = 8 ; threads < 513; threads*=2){
            results <<","<<cyclic_parallel_sum(A,B,C,size,threads);
        }
        for (uint64_t threads = 8 ; threads < 513; threads*=2){
            results <<","<< sum_dynamic(A,B,C,size,threads);
        }
        results << endl;
    }
    //    printResults(C,n,"");
    results.close();
    return 0;
}
