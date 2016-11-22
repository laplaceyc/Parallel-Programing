 /* 
   Sequential Mandelbrot set
 */

#include <X11/Xlib.h>
#include <stdio.h>
#include <unistd.h>

typedef struct complextype
{
	double real, imag;
} Compl;

int main(void)
{
	Display *display;
	Window window;      //initialization for a window
	int screen;         //which screen 

	/* open connection with the server */ 
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		return 0;
	}

	screen = DefaultScreen(display);

	/* set window size */
	int width = 800;
	int height = 800;

	/* set window position */
	int x = 0;
	int y = 0;

	/* border width in pixels */
	int border_width = 0;

	/* create window */
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
					BlackPixel(display, screen), WhitePixel(display, screen));
	
	/* create graph */
	GC gc;
	XGCValues values;
	long valuemask = 0;
	
	gc = XCreateGC(display, window, valuemask, &values);
	//XSetBackground (display, gc, WhitePixel (display, screen));
	XSetForeground (display, gc, BlackPixel (display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
	/* map(show) the window */
	XMapWindow(display, window);
	XSync(display, 0);
	
	/* draw points */
	Compl z, c;
	int repeats;
	double temp, lengthsq;
	int i, j;
	for(i=0; i<width; i++) {
		for(j=0; j<height; j++) {
			z.real = 0.0;
			z.imag = 0.0;
			c.real = ((double)i - 400.0)/200.0; /* Theorem : If c belongs to M(Mandelbrot set), then |c| <= 2 */
			c.imag = ((double)j - 400.0)/200.0; /* So needs to scale the window */
			repeats = 0;
			lengthsq = 0.0;

			while(repeats < 100000 && lengthsq < 4.0) { /* Theorem : If c belongs to M, then |Zn| <= 2. So Zn^2 <= 4 */
				temp = z.real*z.real - z.imag*z.imag + c.real;
				z.imag = 2*z.real*z.imag + c.imag;
				z.real = temp;
				lengthsq = z.real*z.real + z.imag*z.imag; 
				repeats++;
			}

			XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));		
			XDrawPoint (display, window, gc, i, j);
		}
	}
	XFlush(display);
	sleep(5);
	return 0;
}
