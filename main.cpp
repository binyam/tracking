#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <vector>
#define N 10

using namespace std;
using namespace cv;

RNG rng(0xFFFFFFFF);
const int NUMBER = 100;
const int DELAY = 5;
int window_width;
int x_1;
int x_2;
int y_1;
int y_2;


class Blob{
private:
    Point _center;
    Size _axes;
    double _angle;
    Scalar _color;
    bool _occluded;
    vector<Point> _blob_points;
public:
    Blob(){
        _center = Point(100,100);
        _axes = Size(50,30);
        _angle = 0;
        _color = Scalar(255);
        _occluded = false;
    }
    Blob(Point center,
         Size size,
         double angle = 0.0,
         Scalar color = Scalar(0),
         bool occluded = false):
        _center(center),
        _axes(size),
        _angle(angle),
        _color(color),
        _occluded(occluded){
    }

    Blob(const RotatedRect& rect,
         const vector<Point>& points,
         Scalar color,
         bool occluded = false):
        _axes(Size(rect.size.width,rect.size.height)),
        _angle(rect.angle),
        _color(color),
        _blob_points(points),
        _occluded(occluded){
        _center = Point((int)rect.center.x,(int)rect.center.y);
        //cout << _angle << endl;
    }

    void clear(){
        _blob_points.clear();
    }

    Point get_blob_center(){
        return _center;
    }

    Size get_blob_axes(){
        return _axes;
    }

    double get_angle(){
        return _angle;
    }

    void update(Point center){
        _center = center;
    }

    void move(Point delta){
        _center += delta;
    }
    void draw(Mat& image, bool draw_ellipse = false){
        if (!_occluded)
            if (draw_ellipse){
                ellipse(image, _center,_axes,_angle,0,360,_color,2);
            }
            else{
                for(int i = 0; i < _blob_points.size(); i++){
                    Point p(_blob_points[i].y,_blob_points[i].x);
                    if (image.channels() == 1)
                        image.at<uchar>(p.x,p.y) = _color[0];
                    else{
                        image.at<Vec3b>(p.x,p.y)[0] = _color[0];
                        image.at<Vec3b>(p.x,p.y)[1] = _color[1];
                        image.at<Vec3b>(p.x,p.y)[2] = _color[2];
                    }
                }
            }

    }
    void draw(Mat& image, bool draw_ellipse, Scalar color){
        _color = color;
        if (!_occluded)
            if (draw_ellipse){
                ellipse(image, _center,_axes,_angle,0,360,_color,2);
            }
            else{
                for(int i = 0; i < _blob_points.size(); i++){
                    Point p(_blob_points[i].y,_blob_points[i].x);
                    if (image.channels() == 1)
                        image.at<uchar>(p.x,p.y) = _color[0];
                    else{
                        image.at<Vec3b>(p.x,p.y)[0] = _color[0];
                        image.at<Vec3b>(p.x,p.y)[1] = _color[1];
                        image.at<Vec3b>(p.x,p.y)[2] = _color[2];
                    }
                }
            }

    }

};

class Person{
private:
    Blob _face;
    Blob _right_hand;
    Blob _left_hand;
    int _occluded;
public:
    Person(){

        _face = Blob(Point(160,60),
                     Size(30,30),
                     0.0,
                     Scalar(0,255,0));
        _right_hand = Blob(Point(90,160),
                          Size(30,30),
                          0.0,
                          Scalar(0,255,255));
        _left_hand = Blob(Point(250,160),
                           Size(30,30),
                           0.0,
                           Scalar(0,0,255));
    }
    Person(Blob face, Blob right_hand, Blob left_hand):
        _face(face),
        _right_hand(right_hand),
        _left_hand(left_hand){
    }
    void clear(){
        _face.clear();
        _right_hand.clear();
        _left_hand.clear();
    }

    void set_face(Blob face){
        _face = face;
    }
    void set_right_hand(Blob right_hand){
        _right_hand = right_hand;
    }

    void set_left_hand(Blob left_hand){
        _left_hand = left_hand;
    }

    Blob get_face(){
        return _face;
    }

    Blob get_right_hand(){
        return _right_hand;
    }

    Blob get_left_hand(){
        return _left_hand;
    }

    void draw(Mat& image, bool draw_ellipses){
        _face.draw(image,draw_ellipses);
        _right_hand.draw(image,draw_ellipses);
        _left_hand.draw(image,draw_ellipses);
    }
    void move(const Point& delta_FACE,
              const Point& delta_RIGHT_HAND,
              const Point& delta_LEFT_HAND){
        _face.move(delta_FACE);
        _right_hand.move(delta_RIGHT_HAND);
        _left_hand.move(delta_LEFT_HAND);
    }
    void print(){
        cout << _face.get_blob_center() << endl;
    }

};

class Tracker{
private:
    vector<Person> _tracker;
    int _top;
    Point _d_face;
    Point _d_rhand;
    Point _d_lhand;
protected:
    void init_deltas(){
        _d_face = Point(0,0);
        _d_rhand = Point(0,0);
        _d_lhand = Point(0,0);
    }
    void update_deltas(){
        Person p_1, p_2;
        p_1 = _tracker[_top];
        p_2 = _tracker[_top - max(0,_top-1)];

        _d_face = p_1.get_face().get_blob_center() -
                p_2.get_face().get_blob_center();

        _d_rhand = p_1.get_right_hand().get_blob_center() -
                p_2.get_right_hand().get_blob_center();

        _d_lhand = p_1.get_left_hand().get_blob_center() -
                p_2.get_left_hand().get_blob_center();
    }

public:
    Tracker(){
        Person p1;
        _tracker.push_back(p1);
        _top = _tracker.size() - 1;
        init_deltas();
    }

    Tracker(Person p1){
        _tracker.push_back(p1);
        _top = _tracker.size() - 1;
        init_deltas();
    }

//    void predict2(Point df, Point drh, Point dlh){
//        _person.move(df,drh,dlh);
//    }

    void predict(){
        update_deltas();
        Person p1 = _tracker[_top];
        p1.move(_d_face, _d_rhand, _d_lhand);
        _tracker.push_back(p1);
    }

    Person get_hypothesis(){
        return _tracker[_top];
    }

    void draw(Mat& image, bool draw_ellipses){
        if (_top >= 0)
            _tracker[_top].draw(image, draw_ellipses);
    }

    void print(){
        _tracker[_top].print();
    }

    void update(Person p){
        // checking can be done here
        _tracker[_top] = p;
    }
};


class ColorHistogram{
private:
    int _hist_size[3]; // Number of bins
    float _hranges[2];
    const float* _ranges[3];
    int _channels[3];
public:
    ColorHistogram(){
        _hist_size[0] = 256; // 180 for hue
        _hist_size[1] = 256;
        _hist_size[2] = 256;

        _hranges[0] = 0.0;
        _hranges[1] = 255.0;

        _ranges[0] = _hranges;
        _ranges[1] = _hranges;
        _ranges[2] = _hranges;

        _channels[0] = 0;
        _channels[1] = 1;
        _channels[2] = 2;
    }

    MatND getHistogram(const Mat& image, const Mat& mask = Mat()){
        MatND hist;
        calcHist(&image,
                 1,
                 _channels,
                 mask,
                 hist,
                 1,
                 _hist_size,
                 _ranges);
        return hist;
    }

    SparseMat getSparseHistogram(const Mat& image, const Mat& mask = Mat()){
        SparseMat hist(3,_hist_size,CV_32F);
        calcHist(&image,
                 1,
                 _channels,
                 mask,
                 hist,
                 3,
                 _hist_size,
                 _ranges);
        return hist;
    }

    MatND getHueHistogram(const Mat& image,const Mat& mask = Mat()){
        MatND hist;

        Mat hue;
        cvtColor(image,hue,CV_BGR2HSV);

        _hranges[0] = 0.0;
        _hranges[1] = 180.0;
        _channels[0] = 0;

        calcHist(&hue,
                 1,
                 _channels,
                 mask,
                 hist,
                 1,
                 _hist_size,
                 _ranges);
        return hist;
    }

    Mat colorReduce(const Mat& image, int div = 64){
        int shift_size = static_cast<int>(log(static_cast<double>(div))/log(2.0));
        uchar mask = 0xFF << shift_size;

        Mat_<Vec3b>::const_iterator it = image.begin<Vec3b>();
        Mat_<Vec3b>::const_iterator itend = image.end<Vec3b>();

        Mat result(image.rows, image.cols, image.type());
        Mat_<Vec3b>::iterator itr = result.begin<Vec3b>();

        for( ; it != itend; it++, ++itr){
            (*itr)[0]= ((*it)[0] & mask) + div/2;
            (*itr)[1]= ((*it)[1] & mask) + div/2;
            (*itr)[2]= ((*it)[2] & mask) + div/2;
        }
        return result;
    }

};

class watershedSegmenter{
private:
    Mat markers;
public:
    void setMarkers(const Mat& markerImage){
        markerImage.convertTo(markers, CV_32S);
    }

    Mat process(const Mat& image){
        watershed(image, markers);
        return markers;
    }

    Mat getSegmentation(){
        Mat temp;
        markers.convertTo(temp, CV_8U);
        return temp;
    }

    Mat getWatersheds(){
        Mat temp;
        markers.convertTo(temp, CV_8U, 255,255);
        return temp;
    }


};


int main_x(int argc, char* argv[]){

    string window_name = "Tracking";
    Mat image = imread("/Users/bingeb/avatech/data/peter1.jpg");
    Mat frame, output;
    VideoCapture cap("/Users/bingeb/clara/data/polish_sl.flv");

    Mat gray, skin;

    BackgroundSubtractorMOG mog;

    namedWindow(window_name,1);
    //setMouseCallback(window_name,onMouse);
    while(true){
        cap >> frame;
        if (frame.empty()) break;
        if(waitKey(30) > 0)
            break;
          cvtColor(frame,gray,CV_BGR2GRAY);

          adaptiveThreshold(gray,
                            gray,
                            255,
                            ADAPTIVE_THRESH_GAUSSIAN_C,
                            CV_THRESH_BINARY,
                            115,0);

//        mog(frame,foreground,0.01);
//        threshold(foreground,foreground, 128,255,THRESH_BINARY + THRESH_OTSU);
       imshow("Display", gray);
    }

    waitKey();

    return 0;
}

int main_cr(int argc, char* argv[]){

    Mat image = imread("/Users/bingeb/avatech/data/peter1.jpg");
    ColorHistogram h;
    MatND hist = h.getHistogram(image);
    normalize(hist,hist,1.0);

//    calcBackProject(&image,
//                    1,
//                    channels,
//                    histogram,
//                    result,
//                    ranges,
//                    255.0);
    imshow("Display", h.colorReduce(image,32));
    waitKey();
    cout << "hello world!" << endl;
    return 0;
}


Mat binarize(const Mat& frame){

    Mat gray, kernel;
    GaussianBlur(frame,frame,Size(5,5),0);
    cvtColor(frame,gray,CV_BGR2GRAY);
    threshold(gray,gray,80,255,THRESH_BINARY);

    kernel = getStructuringElement(MORPH_RECT,Size(5,5));
    dilate(gray,gray,kernel,Point(-1,-1),1);

//    adaptiveThreshold(gray,
//                      gray,
//                      255,
//                      ADAPTIVE_THRESH_GAUSSIAN_C,
//                      CV_THRESH_BINARY,
//                      115,-30);
    return gray;
}

bool is_good(vector<Point> contour){
// Eliminate too short or too long contours
// other constraints can be added here
    int cmin = 100;
    int cmax = 1000;
    return (contour.size() > cmin && contour.size() < cmax);
}

vector <vector<Point> > get_blobs(const Mat& frame, const MatND& hist){
    Mat binary_image = binarize(frame);
    vector<vector<Point> > contours;
    findContours(binary_image,
                 contours, // a vector of contours
                 CV_RETR_EXTERNAL, // retrieve the external contours
                 CV_CHAIN_APPROX_NONE); // all pixels of each contours

    vector<vector<Point> > :: iterator itc = contours.begin();
    while(itc != contours.end()){
        if (!is_good(*itc)){
            itc = contours.erase(itc);
        }
        else
            ++itc;
    }
    return contours;
}

double dist(Blob b, Point p){

    Point center = b.get_blob_center();
    Size axes = b.get_blob_axes();

    double c = (p.x - center.x)/((double)axes.width);

    double d = (p.y - center.y)/((double)axes.height);

    double ad = b.get_angle();

    double ar = M_PI*ad/180.0; // change to radian


    double t1 = c*cos(ar) - d*sin(ar);
    double t1_2 = t1 * t1;

    double t2 = c*sin(ar) + d*cos(ar);
    double t2_2 = t2 * t2;

    return sqrt(t1_2 + t2_2);
}


void mark(Mat& image, const Point& p,const Scalar& color){
    int i = p.y;
    int j = p.x;
    image.at<Vec3b>(i,j)[0] = color[0];
    image.at<Vec3b>(i,j)[1] = color[1];
    image.at<Vec3b>(i,j)[2] = color[2];
}

Person apply_association_rules(Person hypothesis, Mat& image, Mat& frame){

    Blob h_face = hypothesis.get_face();
    Blob h_right_hand = hypothesis.get_right_hand();
    Blob h_left_hand = hypothesis.get_left_hand();

//    h_face.draw(frame,true,Scalar(0,0,0));
//    h_right_hand.draw(frame,true, Scalar(255,255,255));
//    h_left_hand.draw(frame,true, Scalar(255,0,0));

    vector<Point> face_points;
    vector<Point> right_hand_points;
    vector<Point> left_hand_points;

    // to remember: write code to deal with joined hands that split
    // a hypothesis shared by more than one blob
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++){
            if(image.at<uchar>(i,j) == 255){
                Point p(j,i);
                int intensity = 255;
               // double alpha = 0.5;
                bool in_on = false;
                Scalar color = Scalar::all(0);
                double d1 = dist(h_face,p);
                if (d1 <= 1.0){
                    face_points.push_back(p);
                    in_on = true;
                    color[1] =  intensity;
                }

                double d2 = dist(h_right_hand,p);
                if (d2 <= 1.0){
                    right_hand_points.push_back(p);
                    in_on = true;
//                    if(color[1]){
//                        color[1] = alpha*color[1]+ (1-alpha)*intensity;
//                        color[2] = alpha*color[2]+ (1-alpha)*intensity;
//                    }
//                    else
                    color[1] = color[2] = intensity;
                }

                double d3 = dist(h_left_hand,p);
                if (d3 <= 1.0){
                    left_hand_points.push_back(p);
                    in_on = true;
                    color[2] += intensity;
                }

                if (!in_on){
                    if (d1 < d2 &&  d1 < d3){
                        face_points.push_back(p);
                        color[1] += intensity;
                    }
                    if (d2 < d1 && d2 < d3){
                        right_hand_points.push_back(p);
                        color[1] += intensity;
                        color[2] += intensity;
                    }
                    if (d3 < d1 && d3 < d2){
                        left_hand_points.push_back(p);
                        color[2] += intensity;
                    }
                //    else{}
                }
                mark(frame,p,color);

            }
        }

    Blob face(h_face);
    Blob left_hand(h_left_hand);
    Blob right_hand(h_right_hand);

    if(face_points.size() >= 5){
        RotatedRect rect_face = fitEllipse(Mat(face_points));
        Blob f(rect_face,face_points,Scalar(0,255,0));
        face = f;
    }
    if(right_hand_points.size() >= 5){
        RotatedRect rect_right_hand = fitEllipse(Mat(right_hand_points));
        Blob rh(rect_right_hand, right_hand_points,Scalar(0,255,255));
        right_hand = rh;
    }
    if(left_hand_points.size() >= 5){
        RotatedRect rect_left_hand = fitEllipse(Mat(left_hand_points));
        Blob lh(rect_left_hand, left_hand_points,Scalar(0,0,255));
        left_hand = lh;
    }

    Person p(face,right_hand,left_hand);

    return p;
}
static Scalar randomColor( RNG& rng )
{
    int icolor = (unsigned) rng;
    return Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
}

int Displaying_Random_Text( Mat image, RNG rng )
{
  int lineType = 8;

  for ( int i = 1; i < NUMBER; i++ )
  {
    Point org;
    org.x = rng.uniform(x_1, x_2);
    org.y = rng.uniform(y_1, y_2);

    putText( image, "Dear Peter! Happy New Year", org, rng.uniform(0,8),
             rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);
  }

  return 0;
}

int Displaying_Big_End( Mat image, RNG )
{
  Size textsize = getTextSize(" New Year!", CV_FONT_HERSHEY_COMPLEX, 3, 5, 0);
  Point org((image.cols - textsize.width)/2, (image.rows - textsize.height)/2);
  int lineType = 8;

  Mat image2;

  for( int i = 0; i < 255; i += 2 )
  {
    //image2 = image - Scalar::all(i);
    putText( image, "Dear Peter!", org, CV_FONT_HERSHEY_COMPLEX, 2,
             Scalar(255, i, 255), 4, lineType );
    putText( image, "Happy New Year!", Point(org.x,org.y*2), CV_FONT_HERSHEY_COMPLEX, 2,
             Scalar(255, i, 255), 4, lineType );
  }
  //image2.copyTo(image);
  return 0;
}
// A view requests from the model
// the information that it needs to generate an output representation
// mvc
// model - data, representation, object
// view - presentation, display, write
// controller - get input from user/file, talk to model, talk to view
int main(int argc, char* argv[]){

    Mat frame, image, dst;
    VideoWriter writer;
    VideoCapture cap;
    string win_name = "Tracking";

    if(!cap.open("/Users/bingeb/clara/data/SSL_JM_poem_cayak.mpg"))
            return -1;

    double fps = cap.get(CV_CAP_PROP_FPS);
    int width = static_cast<int> (cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int> (cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    writer.open("happy2013.avi",
                CV_FOURCC('D', 'I', 'V', 'X'),
                fps,
                Size(width*2, height));

    Mat result = Mat(height,width*2, CV_8UC3);
    window_width = width*2;

    x_1 = -window_width/2;
    x_2 = window_width*3/2;
    y_1 = -window_width/2;
    y_2 = window_width*3/2;

    Mat left = result(Range(0,height),Range(0,width));
    Mat right = result(Range(0, height),Range(width,2*width));

    Person hypothesis, person, person_prev;

    namedWindow(win_name);

    Mat blobs;
    bool with_ellipse = true;

    Point df(0,0),drh(0,0),dlh(0,0);
    int i = 0;
    char key;
    Displaying_Big_End(result, rng );
    for(int i = 0; i< 25;i++){

    writer << result;
    }
    while(true){
        cout << i++ << '\t';
        cap >> frame;
        frame.copyTo(image);
        blur(frame,frame,Size(5,5));

        if(frame.empty())
            break;
        key = waitKey(10);

        if((int)key == 27)
            break;
        if(key == ' ')
            waitKey();

        // generates hypotheses (predicts person (face and hands) config
        //tracker.predict();
//        hypothesis.draw(frame,with_ellipse);
//        waitKey(20);
        hypothesis.move(df,drh,dlh);
//        hypothesis.draw(frame,with_ellipse);

        // gets blobs or contours
        // blobs = get_blobs(frame,hist); // gets blobs from the next frame
        blobs = binarize(frame);


        //  associates prediction with measurement
        person = apply_association_rules(hypothesis, blobs,frame);
        person.draw(frame, with_ellipse);


        df = person.get_face().get_blob_center() -
                person_prev.get_face().get_blob_center();
        drh = person.get_right_hand().get_blob_center() -
                person_prev.get_right_hand().get_blob_center();
        dlh = person.get_left_hand().get_blob_center() -
                person_prev.get_left_hand().get_blob_center();

        person.clear();
        person_prev = person;
        //imshow(win_name,frame);
        image.copyTo(left);
        frame.copyTo(right);
        //addWeighted( left, 0.5, right, 0.5, 0.0, dst);
        //Displaying_Random_Text( result, rng );
        //Displaying_Big_End(result, rng );
        imshow(win_name,result);
        writer << result;
        cout << endl;
    }
    Displaying_Big_End(result, rng );
    imshow(win_name,result);
    writer << result;
}

void timing(){
    int64 tinit = getTickCount();
    double tpermsec = getTickFrequency();


    cout << (getTickCount()-tinit)/tpermsec << endl;
}
