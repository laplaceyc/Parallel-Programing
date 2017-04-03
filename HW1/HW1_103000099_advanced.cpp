#include <cstdio>
#include <mpi.h>
#include <cstdlib>
#include <climits>
#include <vector>
#include <algorithm>

int main(int argc, char *argv[]) {
    int m, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &m);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 4) {
        if (rank == 0) printf("Wrong parameters. Terminating.\n");
        MPI_Finalize();
        return 0;
    }
    int n = atoi(argv[1]);

    // Input START
    MPI_File in;
    int error = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in);
    if (error) {
        if (rank == 0) printf("Input file read error. Terminating.\n");
        MPI_Finalize();
        return 0;
    }
    int *buf = new int[n];
    std::vector<int> v_buf;
    MPI_File_read(in, buf, n, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&in);
    // Input END

    // Sort START
    long long min = buf[0];
    long long max = buf[0];
    for (int i = 1; i < n; i++) {
        if (buf[i] < min) min = buf[i];
        if (buf[i] > max) max = buf[i];
    }
    long long bucket_interval_length = (max - min) / m; // HACK when m = 1
    int lower = (int) (min + bucket_interval_length * rank);
    int upper = rank == m - 1 ? (int) max : (int) (lower + bucket_interval_length - 1);
    for (int i = 0; i < n; i++) {
        if (buf[i] >= lower && buf[i] <= upper) {
            v_buf.push_back(buf[i]);
        }
    }
    std::sort(v_buf.begin(), v_buf.end());
    // Sort END
    delete [] buf;

    // Output START
    int accumulate_size = 0;
    if (rank != 0) MPI_Recv(&accumulate_size, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    accumulate_size += v_buf.size();
    if (rank != m - 1) MPI_Send(&accumulate_size, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    MPI_File out;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &out);
    MPI_File_write_at(out, (accumulate_size - v_buf.size()) * sizeof(int), v_buf.data(), v_buf.size(), MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&out);
    // Output END
    MPI_Finalize();
    return 0;
}
