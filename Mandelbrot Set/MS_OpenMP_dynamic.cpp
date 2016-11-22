#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#include <complex>
#include <string.h>
#include <X11/Xlib.h>
#define MAX_ITERATION 100000

using namespace std;

int cal_pixel(complex<double> c){
	int k;
//	double len;
	complex<double> z(0, 0);
	for(k = 0; k < MAX_ITERATION; k++){
		z = z * z + c;
		if((z.real() * z.real() + z.imag() * z.imag()) > 4)	break;
	}
	return k;
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

	//calculate complex c
	complex<double> c;
	double x_scale = (RIGHT - LEFT) / POINT_NUM_X;
    double y_scale = (UPPER - LOWER) / POINT_NUM_Y;
	int result;    	
	int i, j;
	omp_lock_t mylock;
	if(ENABLE_XWINDOW){ 
		Xwindow(POINT_NUM_X, POINT_NUM_Y);
		omp_init_lock(&mylock);
	}
	#pragma omp parallel num_threads(THREAD_NUM) private(i, j, c, result)
    {
        #pragma omp for schedule(dynamic, 1)
        for(i = 0; i < POINT_NUM_Y; i++) {
            for(j = 0; j < POINT_NUM_X; j++) {
                c = complex<double>(LEFT + x_scale * j,LOWER + y_scale * i);
				result = cal_pixel(c);
				if(ENABLE_XWINDOW){
					omp_set_lock(&mylock);
					XSetForeground(display, gc, 1024*1024*(result%256));
					XDrawPoint(display, window, gc, j, i);
					omp_unset_lock(&mylock);
				}
			}
        }
		if(ENABLE_XWINDOW) omp_destroy_lock(&mylock);
	}
	
	if(ENABLE_XWINDOW) {
		XFlush(display); 
		sleep(5);
	}

    return 0;
}
	
