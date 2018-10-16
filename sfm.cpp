#define CERES_FOUND true
#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::sfm;

static void help() {
    cout
        << "\n------------------------------------------------------------------\n"
        << " This program shows the camera trajectory reconstruction capabilities\n"
        << " in the OpenCV Structure From Motion (SFM) module.\n"
        << " \n"
        << " Usage:\n"
        << "        example_sfm_trajectory_reconstruction <path_to_tracks_file> <f> <cx> <cy>\n"
        << " where: is the tracks file absolute path into your system. \n"
        << " \n"
        << "        The file must have the following format: \n"
        << "        row1 : x1 y1 x2 y2 ... x36 y36 for track 1\n"
        << "        row2 : x1 y1 x2 y2 ... x36 y36 for track 2\n"
        << "        etc\n"
        << " \n"
        << "        i.e. a row gives the 2D measured position of a point as it is tracked\n"
        << "        through frames 1 to 36.  If there is no match found in a view then x\n"
        << "        and y are -1.\n"
        << " \n"
        << "        Each row corresponds to a different point.\n"
        << " \n"
        << "        f  is the focal lenght in pixels. \n"
        << "        cx is the image principal point x coordinates in pixels. \n"
        << "        cy is the image principal point y coordinates in pixels. \n"
        << "------------------------------------------------------------------\n\n"
        << endl;
}


/* Build the following structure data
 *
 *            frame1           frame2           frameN
 *  track1 | (x11,y11) | -> | (x12,y12) | -> | (x1N,y1N) |
 *  track2 | (x21,y11) | -> | (x22,y22) | -> | (x2N,y2N) |
 *  trackN | (xN1,yN1) | -> | (xN2,yN2) | -> | (xNN,yNN) |
 *
 *
 *  In case a marker (x,y) does not appear in a frame its
 *  values will be (-1,-1).
 */

void
parser_2D_tracks(const string &_filename, std::vector<Mat> &points2d)
{
    ifstream myfile(_filename.c_str());

    if (!myfile.is_open())
    {
        cout << "Unable to read file: " << _filename << endl;
        exit(0);
    } else {

        double x, y;
        string line_str;
        int n_frames = 0, n_tracks = 0;

        // extract data from text file

        vector<vector<Vec2d> > tracks;
        for ( ; getline(myfile,line_str); ++n_tracks)
        {
            istringstream line(line_str);

            vector<Vec2d> track;
            for ( n_frames = 0; line >> x >> y; ++n_frames)
            {
                if ( x > 0 && y > 0)
                    track.push_back(Vec2d(x,y));
                else
                    track.push_back(Vec2d(-1));
            }
            tracks.push_back(track);
        }

        // embed data in reconstruction api format

        for (int i = 0; i < n_frames; ++i)
        {
            Mat_<double> frame(2, n_tracks);

            for (int j = 0; j < n_tracks; ++j)
            {
                frame(0,j) = tracks[j][i][0];
                frame(1,j) = tracks[j][i][1];
            }
            points2d.push_back(Mat(frame));
        }
        myfile.close();
    }
}


//Keyboard callback to control 3D visualization
bool camera_pov = false;

void keyboard_callback(const viz::KeyboardEvent &event, void* cookie)
{
    if ( event.action == 0 &&!event.symbol.compare("s") )
        camera_pov = !camera_pov;
}

int main(int argc, char** argv)
{
    if ( argc != 5 )
    {
        help();
        exit(0);
    }

    // Read 2D points from text file
    std::vector<Mat> points2d;
    parser_2D_tracks( argv[1], points2d );

    // Set the camera calibration matrix
    const double f = atof(argv[2]),
          cx = atof(argv[3]), cy = atof(argv[4]);

    Matx33d K = Matx33d(f, 0, cx,
            0, f, cy,
            0, 0,  1);

    /// Reconstruct the scene using the 2d correspondences
    bool is_projective = true;
    vector<Mat> Rs_est, ts_est, points3d_estimated;
    reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, is_projective);

    // Print output
    cout << "\n----------------------------\n" << endl;
    cout << "Reconstruction: " << endl;
    cout << "============================" << endl;
    cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
    cout << "Estimated cameras: " << Rs_est.size() << endl;
    cout << "Refined intrinsics: " << endl << K << endl << endl;

    cout << "3D Visualization: " << endl;
    cout << "============================" << endl;

    /// Create 3D windows
    viz::Viz3d window("Coordinate Frame");
    window.setWindowSize(Size(500,500));
    window.setWindowPosition(Point(150,150));
    window.setBackgroundColor(); // black by default

    // Create the pointcloud
    cout << "Recovering points  ... ";

    // recover estimated points3d
    vector<Vec3f> point_cloud_est;
    for (unsigned int i = 0; i < points3d_estimated.size(); ++i)
        point_cloud_est.push_back(Vec3f(points3d_estimated[i]));

    cout << "[DONE]" << endl;
    double f_refined, cx_refined, cy_refined;
    f_refined = K(0,0);
    cx_refined = K(0,2);
    cy_refined = K(1,2);

    FileStorage fs("points3d.txt", FileStorage::WRITE);
    fs << "points3d" << point_cloud_est;
    fs << "Rs_est" << Rs_est;
    fs << "ts_est" << ts_est;
    fs << "f" << f_refined;
    fs << "cx" << cx_refined;
    fs << "cy" << cy_refined;

    /// Recovering cameras
    cout << "Recovering cameras ... ";

    vector<Affine3d> path;
    for (size_t i = 0; i < Rs_est.size(); ++i)
        path.push_back(Affine3d(Rs_est[i],ts_est[i]));

    cout << "[DONE]" << endl;


    /// Add the pointcloud
    if ( point_cloud_est.size() > 0 )
    {
        cout << "Rendering points   ... ";

        viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
        window.showWidget("point_cloud", cloud_widget);

        cout << "[DONE]" << endl;
    }
    else
    {
        cout << "Cannot render points: Empty pointcloud" << endl;
    }

    /// Add cameras
    if ( path.size() > 0 )
    {
        cout << "Rendering Cameras  ... ";

        //window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
        //window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::yellow()));
        //window.setViewerPose(path[0]);

        cout << "[DONE]" << endl;
    }
    else
    {
        cout << "Cannot render the cameras: Empty path" << endl;
    }

    /// Wait for key 'q' to close the window
    cout << endl << "Press 'q' to close each windows ... " << endl;

    window.spin();

    return 0;
}
