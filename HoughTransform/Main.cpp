
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <vector>
#include <iostream>
#include <assert.h>
#include "Windows.h"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;

#define IMG_FILENAME "data\\Bob.jpg"

#define BLACK 0, 0, 0
#define WHITE 255, 255, 255
#define RED 0, 0, 255
#define GREEN 0, 255, 0
#define BLUE 255, 0, 0

#define YX_OUT cout << y << ", " << x << endl

#define WIDTH 500
#define HEIGHT 500

#define PIXEL_SIZE 5
#define PRECISION 0.05

bool RUNNING = true;

////////////////////////////////////////////////////////////////

int addVec3b(const Vec3b a) {
	return a[0] + a[1] + a[2];
}

int max_(const int a, const int b) {
	if (a > b)
		return a;
	return b;
}

int min_(const int a, const int b) {
	if (a < b)
		return a;
	return b;
}

struct Line {
	int y0;
	int x0;
	int y1;
	int x1;
};

struct LineParams {
	double theta;
	double rho;
	int i_theta;
	int i_rho;
};

////////////////////////////////////////////////////////////////

class Hough {
private:
	int max_value;
	double max_theta, max_rho;
	int max_i_theta, max_i_rho;

	int max_diameter;

	Mat hough_points;
	Mat hough_img;

	vector<Line> lines;
	vector<LineParams> lines_params;
public:
	Hough() {
		init();
	}

	void init() {
		max_value = 0;
		max_theta = -1;
		max_rho = -1;
		max_i_theta = -1;
		max_i_rho = -1;

		max_diameter = (int)(sqrt(pow(WIDTH, 2) + pow(HEIGHT, 2)));

		hough_points = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
		hough_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

		lines.clear();
		lines_params.clear();
	}

	void calcHoughLine(const int y, const int x, const int channel) {
		double theta, rho;
		int i_theta, i_rho;
		int value;
		char grey_value;
		Vec3b prev_values;
		for (theta = -90.; theta < 90.; theta += PRECISION) {
			rho = x * cos(theta * M_PI / 180) + y * sin(theta * M_PI / 180);

			convertToImgSpace(theta, rho, &i_theta, &i_rho);

			prev_values = hough_points.at<Vec3b>(i_rho, i_theta);
			if (channel == -1) {
				for (int c = 0; c < 3; c++) {
					hough_points.at<Vec3b>(i_rho, i_theta)[c]++;
				}
			}
			else {
				hough_points.at<Vec3b>(i_rho, i_theta)[channel]++;
			}
			for (int c = 0; c < 3; c++) {
				if (hough_points.at<Vec3b>(i_rho, i_theta)[c] < prev_values[c]) {
					hough_points.at<Vec3b>(i_rho, i_theta)[c] = 255;
					createLine(theta, rho, i_theta, i_rho);
				}
			}
			
			value = addVec3b(hough_points.at<Vec3b>(i_rho, i_theta));

			if (value > max_value) {
				max_value = value;
				max_theta = theta;
				max_rho = rho;
				max_i_theta = i_theta;
				max_i_rho = i_rho;
			}

			grey_value = min_((int)(log(value) * 20), 255);
			if (channel == -1)
				hough_img.at<Vec3b>(i_rho, i_theta) = Vec3b(grey_value, grey_value, grey_value);
			else
				hough_img.at<Vec3b>(i_rho, i_theta)[channel] = grey_value;
		}
	}

	const void convertToImgSpace(const double theta, const double rho, int* i_theta, int* i_rho) {
		*i_theta = (int)(((theta + 90.) / 180.) * WIDTH);
		*i_rho = (int)(((rho + max_diameter) / (2 * max_diameter)) * HEIGHT);
	}

	bool createLine(const double theta, const double rho, const int i_theta, const int i_rho) {
		if (i_theta == -1 && i_rho == -1)
			return false;
		for (LineParams lp : lines_params) {
			if (lp.theta == theta && lp.rho == rho && lp.i_theta == i_theta && lp.i_rho == i_rho) {
				return false;
			}
		}

		Line l = getLine(theta, rho, i_theta, i_rho);

		lines_params.push_back(LineParams{ theta, rho, i_theta, i_rho });
		lines.push_back(l);
		return true;
	}

	static Line getLine(const double theta, const double rho, const int i_theta, const int i_rho) {
		if (i_theta == -1 && i_rho == -1)
			return Line{ -1, -1, -1, -1 };

		int x0, y0, x1, y1;

		x0 = 0;
		x1 = WIDTH;
		y0 = (int)((rho - (x0 * cos(theta * M_PI / 180))) / sin(theta * M_PI / 180));
		y1 = (int)((rho - (x1 * cos(theta * M_PI / 180))) / sin(theta * M_PI / 180));

		if (y0 < -(pow(10, 5)) || y1 >(pow(10, 5))) {
			y0 = 0;
			y1 = HEIGHT;

			x0 = (int)((rho - (y0 * sin(theta * M_PI / 180))) / cos(theta * M_PI / 180));
			x1 = (int)((rho - (y1 * sin(theta * M_PI / 180))) / cos(theta * M_PI / 180));
		}

		return Line{ y0, x0, y1, x1 };
	}

	vector<Line> getLines() {
		return lines;
	}

	Line getMaxLine() {
		return getLine(max_theta, max_rho, max_i_theta, max_i_rho);
	}

	const Mat getImage() {
		return hough_img;
	}

	const Mat getMaxImage() {
		Mat max_hough_img = hough_img.clone();

		if (max_i_theta == -1 && max_i_rho == -1)
			return max_hough_img;

		circle(max_hough_img, Point(max_i_theta, max_i_rho), 30, Scalar(RED), 2, 8, 0);

		return max_hough_img;
	}

	void clear() {
		init();
	}
};

////////////////////////////////////////////////////////////////

class Painter {
private:
	Mat canvas;
	Mat canvas_img;
public:
	Painter() {
		init();
	}
	Painter(const String &filename, Hough &hough) {
		loadImage(filename, hough);
	}

	void init() {
		canvas = Mat(HEIGHT/PIXEL_SIZE, WIDTH/PIXEL_SIZE, CV_8UC3, Scalar(WHITE));
		canvas_img = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(WHITE));
	}

	void loadImage(const String &filename, Hough &hough) {
		cout << "Loading Image: '" << filename << "'..." << endl;

		Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);

		Mat small_img;
		Mat small_canny;

		resize(img, small_img, Size(WIDTH / PIXEL_SIZE, HEIGHT / PIXEL_SIZE));
		Canny(small_img, small_canny, 100, 300, 3);

		bitwise_not(small_canny, canvas);
		cvtColor(canvas, canvas, CV_GRAY2BGR);

		resize(canvas, canvas_img, Size(WIDTH, HEIGHT), 0, 0, INTER_NEAREST);

		calcHoughFromImage(hough);
	}

	void calcHoughFromImage(Hough &hough) {
		hough.clear();
		for (int y = 0; y < canvas.rows; y++) {
			for (int x = 0; x < canvas.cols; x++) {
				Vec3b color = canvas.at<Vec3b>(y, x);
				if (color == Vec3b(BLACK)) {
					hough.calcHoughLine(y*PIXEL_SIZE, x*PIXEL_SIZE, -1);
				}
				else if (color == Vec3b(RED))
					hough.calcHoughLine(y*PIXEL_SIZE, x*PIXEL_SIZE, 2);
				else if (color == Vec3b(GREEN))
					hough.calcHoughLine(y*PIXEL_SIZE, x*PIXEL_SIZE, 1);
				else if (color == Vec3b(BLUE))
					hough.calcHoughLine(y*PIXEL_SIZE, x*PIXEL_SIZE, 0);
			}
		}
	}

	const Mat getCanvas() {
		return canvas_img;
	}


	const Mat getMaxLineCanvas(Line &ml) {
		Mat max_canvas_img = canvas_img.clone();

		if (ml.x0 == -1 && ml.y0 == -1 && ml.y1 == -1 && ml.y1 == -1)
			return max_canvas_img;

		line(max_canvas_img, Point(ml.x0, ml.y0), Point(ml.x1, ml.y1), Scalar(RED), 2, 8, 0);

		return max_canvas_img;
	}

	const Mat getLinesCanvas(Line &ml, vector<Line> &lines) {
		Mat lines_canvas_img = canvas_img.clone();

		for (Line l : lines) {
			line(lines_canvas_img, Point(l.x0, l.y0), Point(l.x1, l.y1), Scalar(GREEN), 1, 8, 0);
		}

		if (ml.x0 != -1 && ml.y0 != -1 && ml.y1 != -1 && ml.y1 != -1) {
			line(lines_canvas_img, Point(ml.x0, ml.y0), Point(ml.x1, ml.y1), Scalar(RED), 2, 8, 0);
		}

		return lines_canvas_img;
	}

	const Mat getLinesCanvas(vector<Line> &lines) {
		Mat lines_canvas_img = canvas_img.clone();

		for (Line l : lines) {
			line(lines_canvas_img, Point(l.x0, l.y0), Point(l.x1, l.y1), Scalar(GREEN), 1, 8, 0);
		}

		return lines_canvas_img;
	}

	void setPixel(int y, int x, const Vec3b color) {
		x /= PIXEL_SIZE;
		y /= PIXEL_SIZE;
		canvas.at<Vec3b>(y, x) = color;
		x *= PIXEL_SIZE;
		y *= PIXEL_SIZE;

		for (int i = 0; i < PIXEL_SIZE; i++) {
			for (int j = 0; j < PIXEL_SIZE; j++) {
				canvas_img.at<Vec3b>(y+i, x+j) = color;
			}
		}
	}

	const Vec3b getPixel(int y, int x) {
		return canvas.at<Vec3b>((int)(y / PIXEL_SIZE), (int)(x / PIXEL_SIZE));
	}

	void clear() {
		init();
	}
};

////////////////////////////////////////////////////////////////

struct Params {
	Painter &painter;
	Hough &hough;
};

void onMouse(int event, int x, int y, int flags, void* params){

	if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT)
		return;

	Params *mp = (Params*) params;
	Painter &painter = mp->painter;
	Hough &hough = mp->hough;

	if (event == EVENT_LBUTTONDOWN || (event == EVENT_MOUSEMOVE /*&& flags == EVENT_FLAG_LBUTTON*/)) {
		if (painter.getPixel(y, x) == Vec3b(WHITE)) {
			if (flags == EVENT_FLAG_LBUTTON) {
				painter.setPixel(y, x, Vec3b(BLACK));
				hough.calcHoughLine(y, x, -1);
			}
			else if (flags == (EVENT_FLAG_LBUTTON + EVENT_FLAG_CTRLKEY)) {
				painter.setPixel(y, x, Vec3b(RED));
				hough.calcHoughLine(y, x, 2);
			}
			else if (flags == (EVENT_FLAG_LBUTTON + EVENT_FLAG_SHIFTKEY)) {
				painter.setPixel(y, x, Vec3b(GREEN));
				hough.calcHoughLine(y, x, 1);
			}
			else if (flags == (EVENT_FLAG_LBUTTON + EVENT_FLAG_ALTKEY)) {
				painter.setPixel(y, x, Vec3b(BLUE));
				hough.calcHoughLine(y, x, 0);
			}
		}
	}
	else if (event == EVENT_RBUTTONDOWN || (event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_RBUTTON)) {
		if (painter.getPixel(y, x) != Vec3b(WHITE)) {
			//painter.resetPixel(y, x, Vec3b(BLUE));
			//hough.undoHoughLine(y, x, 0);
		}
	}
}

void onKey(char key, void* params) {
	Params *mp = (Params*)params;
	Painter &painter = mp->painter;
	Hough &hough = mp->hough;

	if (key == 'n') {
		painter.clear();
		hough.clear();
	}
	else if (key == 'l') {
		painter.loadImage(IMG_FILENAME, hough);
	}
}

void onExit() {
	cout << "Exiting Program..." << endl;
	RUNNING = false;
}

BOOL WINAPI ConsoleHandler(DWORD dwType) {
	switch (dwType) {
	case CTRL_CLOSE_EVENT:
	case CTRL_LOGOFF_EVENT:
	case CTRL_SHUTDOWN_EVENT:
		onExit();
		return FALSE;
		//return TRUE;
	default:
		break;
	}
	return FALSE;
}

void main() {
	assert(WIDTH%PIXEL_SIZE == 0 && HEIGHT%PIXEL_SIZE == 0);
	SetConsoleCtrlHandler(ConsoleHandler, true);

	Hough hough;
	Painter painter;
	//Painter painter(IMG_FILENAME, hough);

	namedWindow("Canvas", 1);
	moveWindow("Canvas", 0, 0);
	namedWindow("Hough Transformation", 1);
	moveWindow("Hough Transformation", 0, 50 + HEIGHT);

	Params params = { painter, hough };
	setMouseCallback("Canvas", onMouse, (void*)&params);

	int key;
	while (RUNNING) {
		//imshow("Canvas", painter.getMaxLineCanvas(hough.getMaxLine()));
		//imshow("Canvas", painter.getLinesCanvas(hough.getLines()));
		imshow("Canvas", painter.getLinesCanvas(hough.getMaxLine(), hough.getLines()));
		imshow("Hough Transformation", hough.getMaxImage());
		key = waitKey(10);
		if (key != -1)
			onKey(key, (void*)&params);
	}
	destroyAllWindows();
}