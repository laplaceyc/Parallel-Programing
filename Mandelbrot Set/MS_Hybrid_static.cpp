#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <complex>
#include <string.h>
#include <X11/Xlib.h>
#include <omp.h>
#define MAX_ITERATION 100000

using namespace std;

int cal_pixel(complex<double> c){
	int i;
	complex<double> z(0, 0);
	for(i = 0; i < MAX_ITERATION; i++){
		z = z * z + c;
		if((z.real() * z.real() + z.imag() * z.imag()) > 4)	break;
	}
	return i;
}

// global Xwindow parm
Window window;
Display *display;
GC gc;
int screen;

void Xwindow(int width, int height){
	/* open connection with the server */ 
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		return;
	}

	screen = DefaultScreen(display);

	/* set window size */
	//given by main function

	/* set window position */
	int x = 0;
	int y = 0;

	/* border width in pixels */
	int border_width = 0;

	/* create window */
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
					BlackPixel(display, screen), WhitePixel(display, screen));
	
	/* create graph */
	
	XGCValues values;
	long valuemask = 0;
	
	gc = XCreateGC(display, window, valuemask, &values);
	XSetForeground (display, gc, BlackPixel (display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
	/* map(show) the window */
	XMapWindow(display, window);
	XSync(display, 0);
		
    XFlush(display);
	return;
}

int main (int argc, char *argv[]) {
	int rank, size;
	MPI_Comm custom_world = MPI_COMM_WORLD;
	MPI_Group origin_group, new_group;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (argc != 9) {
        fprintf(stderr, "Insuficcient arguments\n");
        fprintf(stderr, "Usage: ./%s N left right lower upper width height enable/disable\n", argv[0]);
        exit(EXIT_FAILURE);
    }
	const int THREAD_NUM = atoi(argv[1]);
	const double LEFT = atof(argv[2]);
	const double RIGHT = atof(argv[3]);
	const double LOWER = atof(argv[4]);
	const double UPPER = atof(argv[5]);
	const int POINT_NUM_X = atoi(argv[6]);
	const int POINT_NUM_Y = atoi(argv[7]);
	int ENABLE_XWINDOW;
	if(strcmp(argv[8],"enable") == 0) ENABLE_XWINDOW = 1;
	else if(strcmp(argv[8],"disable") == 0) ENABLE_XWINDOW = 0;
	else {
		fprintf(stderr, "Error argument: %s\n",argv[8]);
        fprintf(stderr, "Usage: argument must be enable/disable\n");
		exit(EXIT_FAILURE);
	}
	//deal with special case POINT_NUM_X < size
	if (POINT_NUM_X < size) {
		// obtain the group of proc. in the world communicator
		MPI_Comm_group(custom_world, &origin_group);
		// remove unwanted ranks
		int ranges[][3] = {{POINT_NUM_X, size-1, 1}};
		MPI_Group_range_excl(origin_group, 1, ranges, &new_group);
		// create a new communicator
		MPI_Comm_create(custom_world, new_group, &custom_world);
		if (custom_world == MPI_COMM_NULL) {
			// terminate those unwanted processes
			MPI_Finalize();
			exit(0);
		}
		size = POINT_NUM_X;
	}
	//decomposition by x-axis and gain the scale parameter
	int point_per_rank_x = POINT_NUM_X / size;
	int x_start = rank * point_per_rank_x;
	if (rank == size - 1) {//the last rank assigned the remainder (N % size)
		 point_per_rank_x += (POINT_NUM_X % size);
	}
	int x_end = x_start + point_per_rank_x;

	//calculate complex c
	complex<double> c;
	double x_scale = (RIGHT - LEFT) / POINT_NUM_X;
    double y_scale = (UPPER - LOWER) / POINT_NUM_Y;
	int total = POINT_NUM_X * POINT_NUM_Y;
	int *local_buf = (int*)calloc(total, sizeof(int));
    int position;
	int i, j;
	#pragma omp parallel num_threads(THREAD_NUM) private(i, j, c)
	{
		#pragma omp for schedule(static)
		for(i = 0; i < POINT_NUM_Y; i++) {
			for(j = x_start; j < x_end; j++) {
				c = complex<double>(LEFT + x_scale * j,LOWER + y_scale * i);
				position = i * POINT_NUM_X + j;
				local_buf[position] = cal_pixel(c);
			}
		}
	}
	int *root_buf = (int*)calloc(total, sizeof(int));
	if(ENABLE_XWINDOW) MPI_Reduce(local_buf, root_buf, total, MPI_INT, MPI_MAX, size-1, custom_world);
	free(local_buf);
	int m, n;
	//last rank print xwindow
	if(rank == size - 1 && ENABLE_XWINDOW) {
        Xwindow(POINT_NUM_X, POINT_NUM_Y);
        for(m = 0; m < POINT_NUM_Y; m++) {
            for(n = 0; n < POINT_NUM_X; n++) {
                position = m * POINT_NUM_X + n;
                XSetForeground(display, gc, 1024*1024*(root_buf[position]%256));
                XDrawPoint(display, window, gc, n, m);
            }
            XFlush(display);
        }
        sleep(5);
    }
		
    free(root_buf);
	
	MPI_Barrier(custom_world);
    MPI_Finalize();

    return 0;
}
	
