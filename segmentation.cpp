#include "segmentation.h"
#include "WorkerThread.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QQueue>
#include <QFile>
#include <QTextStream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include "Spectra/SymEigsSolver.h"
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/kmeans_plus_plus_initialization.hpp>

//Function Prototypes
Eigen::MatrixXd convertToEigenMat(const QVector<QVector<double> > qtMat);
void symmetryTest(QVector<QVector<double>> &mat);
void writeMatrixToCSV(const QString& filename, const QVector<QVector<double>>& matrix);
QVector<QVector<double>> convertEigenToQVector(const Eigen::MatrixXd& matrix);

QPixmap SegmentImg::segmentImage(QPixmap *pixels){

    if(pixels){
        size_t k = 2;
        QImage image = pixels->toImage();
        unsigned int rows = image.width(); //refers to actual image width
        unsigned int cols = image.height(); //refers to actual image height
        unsigned int new_w = 28;
        unsigned int new_h = 28;

        //calculate adjacency matrix
        QVector<QVector<double>> adjMatrix = createAdjMatrix(image, rows, cols, new_w, new_h);
        std::cout<< "adj matrix success." << std::endl;

        QVector<QVector<double>> degMatrix = createDegreeMatrix(adjMatrix);

        //calculate Laplacian matrix
        QVector<QVector<double>> lapMatrix = createLaplacianMatrix(adjMatrix, degMatrix);
        std::cout<< "Laplacian matrix success." << std::endl;

        //write data to files to then test in matlab:
        std::cout << "Current working directory: " << QDir::currentPath().toStdString() << std::endl;
        // writeMatrixToCSV("./adjMatrix.csv", adjMatrix);
        // writeMatrixToCSV("./degMatrix.csv", degMatrix);
        // writeMatrixToCSV("./lapMatrix.csv", lapMatrix);

        //convert matrix to Eigen lib compatible
        Eigen::MatrixXd lapEigenMat = convertToEigenMat(lapMatrix);
        Eigen::MatrixXd degEigenMat = convertToEigenMat(degMatrix);
        std::cout << "Converted to Eigen compatibility." << std::endl;

        symmetryTest(adjMatrix);
        symmetryTest(lapMatrix);

        bool isSymmetric = true;
        bool isDiagonal = true;
        for (int i = 0; i < degMatrix.size(); i++){
            for (int j = 0; j <degMatrix.size(); j++){
                if (degMatrix[i][j] != degMatrix[j][i]){
                    std::cout << "Matrix is not symmetric." << std::endl;
                    isSymmetric = false;
                    break;
                }
                if (i != j && degMatrix[i][j] != 0) { // Check for non-diagonal elements being non-zero
                    std::cout << "Matrix is not diagonal." << std::endl;
                    isDiagonal = false;
                    break;
                }
            }
        }
        if (isSymmetric) {
            std::cout << "Matrix is symmetric." << std::endl;
        }
        if(isDiagonal){
            std::cout << "Matrix is diagonal." << std::endl;
        }

        /*
        //compute eigenvalues and return all but the first component
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(lapEigenMat); // <--- i believe this is very intensive

        //Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(lapEigenMat, degEigenMat);

        Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors().real();
        std::vector<std::pair<double, Eigen::VectorXd>> eigPairs;
        // Pair eigenvalues with eigenvectors
        for (int i = 0; i < eigenvalues.size(); i++) {
            eigPairs.push_back(std::make_pair(eigenvalues[i], eigenvectors.col(i)));
        }

        // Sort desc pairs based on eigenvalues
        std::sort(eigPairs.begin(), eigPairs.end(),
                  [](const std::pair<double, Eigen::VectorXd> &a, const std::pair<double, Eigen::VectorXd> &b) {
                    return a.first < b.first;
                  });

        // Extract the sorted eigenvectors
        for (int i = 0; i < eigenvalues.size(); i++) {
            eigenvectors.col(i) = eigPairs[i].second;
        }

        for (int i = 0; i < eigenvectors.cols(); i++) {
            eigenvectors.col(i).normalize();
        }
        */

        //ALTERNATIVE TO EIGEN:---------------------------------------------------------------------------------------------
        // Define the matrix operation object using the dense matrix
        Eigen::SparseMatrix<double> sparseMat = lapEigenMat.sparseView();
        Spectra::SparseSymMatProd<double> op(sparseMat);

        // Create the eigen solver object, asking for the 3 smallest eigenvalues
        int numEigenvaluesToCompute = 3; // For example, computing the 3 smallest eigenvalues
        int ncv = std::max(2 * numEigenvaluesToCompute, 20); // ncv is the number of Lanczos vectors, adjust as needed
        Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, numEigenvaluesToCompute, ncv);

        // Initialize and compute
        eigs.init();
        try {
            // Your code to set up and use the compute function
            eigs.compute(Spectra::SortRule::SmallestAlge, 10000, 1e-5);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << std::endl;
            // Handle error
        }
        catch (const std::exception& e) {
            std::cerr << "An exception occurred: " << e.what() << std::endl;
            // Handle error
        }


        // Check for convergence
        Eigen::MatrixXd evectorsTransposed;
        if(eigs.info() == Spectra::CompInfo::Successful)
        {
            Eigen::VectorXd evalues = eigs.eigenvalues();
            Eigen::MatrixXd evectors = eigs.eigenvectors();

            // Use 'evalues' and 'evectors' as needed
            evectors = evectors.middleCols(1,k);
            evectorsTransposed = evectors.transpose();
        }
        else if (eigs.info() == Spectra::CompInfo::NotConverging)
        {
            // Computation did not converge within the maximum number of iterations
            std::cout << "Did not converge towards tolerance" << std::endl;
        }
        else if (eigs.info() == Spectra::CompInfo::NumericalIssue)
        {
            // Numerical issue occurred during computation
        }
        std::cout << "number of iterations reached: "<< eigs.num_iterations() << std::endl;

        arma::mat armaMat(evectorsTransposed.data(),evectorsTransposed.rows(),evectorsTransposed.cols(), false, true);

        /*
        Eigen::MatrixXd kEigenvectors = eigenvectors.middleCols(1,k); //lets test by using just k, not k+1 (k+1 worked well for 0 mnist, but not 7 mnist)
        kEigenvectors.transposeInPlace();
        std::cout << "Eigenvectors Success." << std::endl;
        */
        // empty matrix check
        //arma::mat armaMat(kEigenvectors.data(),kEigenvectors.rows(),kEigenvectors.cols(),false,true);
        if (armaMat.empty()){
            std::cout << "MATRIX IS EMPTY." << std::endl;
        } else {
            std::cout << "Matrix is not empty." << std::endl;
            std::cout << "Matrix is " << armaMat.n_rows << "x" << armaMat.n_cols << std::endl;
        }
        QVector<QVector<double>> keigs = convertEigenToQVector(evectorsTransposed);
        writeMatrixToCSV("./eigMatrix.csv", keigs);


        mlpack::KMeans<mlpack::EuclideanDistance,mlpack::RefinedStart ,mlpack::MaxVarianceNewCluster,mlpack::NaiveKMeans> kmeans(10000);
        arma::Row<size_t> assignments;
        arma::mat centroids;
        std::cout << "beginning KMeans" << std::endl;
        kmeans.Cluster(armaMat, k, assignments, centroids);
        std::cout << "KMeans Success." << std::endl;

        // Create a corresponding arma::Mat<double>
        arma::Mat<double> convertedAssignments(1, assignments.n_elem); // 1 row, same number of columns as 'assignments'

        // Convert and copy each element
        for (size_t i = 0; i < assignments.n_elem; ++i) {
            convertedAssignments(0, i) = static_cast<double>(assignments(i));
        }

        arma::mat reshapedClusters = arma::reshape(convertedAssignments, new_w, new_h);
        std::cout << "Matrix reshaped Success." << std::endl;

        QVector<QColor> colors = {
            QColor(255, 0, 0),    // Red
            QColor(0, 255, 0),    // Green
            QColor(0, 0, 255),    // Blue
            // ... add more colors, one for each cluster
        };


        //convert to pixmap and return
        QImage seg_image(new_w, new_h, QImage::Format_RGB32);
        std::cout << "Convert to Pixmap Success." << std::endl;

        //assign color values to clusters
        for (int x = 0; x < new_w; ++x){
            for (int y = 0; y < new_h; ++y){
                int clusterIndex = reshapedClusters(y,x);
                QColor color = colors.at(clusterIndex % colors.size());//map to a color
                seg_image.setPixel(x, y, color.rgb());
            }
        }


        // Define the scaling factor
        double scaleFactor = 6.0;

        // Calculate the new dimensions
        int scaledWidth = static_cast<int>(new_w * scaleFactor);
        int scaledHeight = static_cast<int>(new_h * scaleFactor);

        // Resize (scale) the image
        QImage scaledImage = seg_image.scaled(scaledWidth, scaledHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);

        QPixmap seg_pixmap = QPixmap::fromImage(scaledImage);
        std::cout << "Segmentation Function End." << std::endl;
        return seg_pixmap;

    }else{
        QPixmap empty;
        return empty;
    }
}

void writeMatrixToCSV(const QString& filename, const QVector<QVector<double>>& matrix) {
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        std::cerr << "Failed to open file for writing: " << filename.toStdString() << std::endl;
        return;
    }

    QTextStream out(&file);

    for (const QVector<double>& row : matrix) {
        QStringList strList;
        for (double val : row) {
            strList << QString::number(val);
        }
        out << strList.join(",") << "\n";
    }
    file.close();
    std::cout << ".csv created." << std::endl;
}


void symmetryTest(QVector<QVector<double>> &mat){
    //symmetry test
    bool isSymmetric = true;
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat.size(); j++) { // Only need to check half the matrix
            if (mat[i][j] != mat[j][i]) {
                isSymmetric = false;
                break;
            }
        }
        if (!isSymmetric) {
            break;
        }
    }
    if (isSymmetric) {
        std::cout << "Matrix is symmetric." << std::endl;
    } else {
        std::cout << "Matrix is not symmetric." << std::endl;
    }
}

QPixmap SegmentImg::convertToGrayscale(const QPixmap &original) {
    QImage image = original.toImage();

    for (int x = 0; x < image.width(); x++) { //note: the increment operator is a pre-increment
        for (int y = 0; y < image.height(); y++) {
            QColor color = image.pixelColor(x, y);
            int gray = qGray(color.rgb()); // Convert color to grayscale
            image.setPixelColor(x, y, QColor(gray, gray, gray));
        }
    }
    return QPixmap::fromImage(image);
}


// Function to convert Eigen::MatrixXd to QVector<QVector<double>>
QVector<QVector<double>> convertEigenToQVector(const Eigen::MatrixXd& matrix) {
    QVector<QVector<double>> qMatrix;
    qMatrix.reserve(matrix.rows()); // Pre-allocate space for rows

    for (int i = 0; i < matrix.rows(); ++i) {
        QVector<double> row;
        row.reserve(matrix.cols()); // Pre-allocate space for columns

        for (int j = 0; j < matrix.cols(); ++j) {
            row.push_back(matrix(i, j));
        }

        qMatrix.push_back(row);
    }

    return qMatrix;
}


//resize img, create adj matrix
QVector<QVector<double>> SegmentImg::createAdjMatrix(QImage image, unsigned int &rows, unsigned int &cols, unsigned int &w, unsigned int &h){
    QPixmap imgPixmap = QPixmap::fromImage(image); // Convert QImage back to QPixmap
    QPixmap resizedPixmap = imgPixmap.scaled(w,h);
    //QPixmap grayscalePixmap = convertToGrayscale(resizedPixmap); // Convert to grayscale
    //QImage usable_img = grayscalePixmap.toImage(); // Convert back to QImage for processing
    QImage usable_img = resizedPixmap.toImage();

    int epsilon = 2;
    int N = w * h;
    QVector<QVector<double>> adjMatrix(N, QVector<double>(N,0));

    //create image vector
    QVector<QColor> vectorImg(N);
    for (int i = 0; i<w; i++){
        for (int j = 0; j<h; j++){
            QColor color = usable_img.pixelColor(i,j);
            vectorImg[i*h+j] = color;
            std::cout << "Red: " << color.red()
                      << ", Green: " << color.green()
                      << ", Blue: " << color.blue() << std::endl;
        }
    }

    //iterate over each pixel
    for (unsigned int i = 0; i < w; i++){
        for (unsigned int j = 0; j < h; j++){
            int idx = j * w + i;

            unsigned int start_i = (i >= 4) ? i - 4 : 0;
            unsigned int end_i = std::min(i + 4, w);

            unsigned int start_j = (j >= 4) ? j - 4 : 0;
            unsigned int end_j = std::min(j + 4, h);

            //iterate through its current kernel (aka neighborhood)
            for (unsigned int ii = start_i; ii < end_i; ii++){
                for (unsigned int jj = start_j; jj < end_j; jj++){

                    //diagonal optimization
                    if (ii == i && jj <= j){
                        continue;
                    }

                    int neigh_idx = jj * w + ii;

                    QColor color_idx = vectorImg[idx];
                    QColor color_neigh = vectorImg[neigh_idx];

                    /*
                    double distance = static_cast<double>(std::sqrt(std::pow(color_idx.red() - color_neigh.red(), 2) +
                                                std::pow(color_idx.green() - color_neigh.green(), 2) +
                                                std::pow(color_idx.blue() - color_neigh.blue(), 2)));
                    */

                    // Convert to HSV color space
                    color_idx = color_idx.toHsv();
                    color_neigh = color_neigh.toHsv();

                    // Correct handling of circular hue distance
                    double hue_diff = color_idx.hue() - color_neigh.hue();
                    if (hue_diff < -180) hue_diff += 360;
                    else if (hue_diff > 180) hue_diff -= 360;

                    double distance = std::sqrt(hue_diff * hue_diff +
                                                std::pow(color_idx.saturation() - color_neigh.saturation(), 2) +
                                                std::pow(color_idx.value() - color_neigh.value(), 2));

                    //add edge to adj matrix
                    if (distance <= epsilon){
                        adjMatrix[idx][neigh_idx] = epsilon;
                        adjMatrix[neigh_idx][idx] = epsilon;
                    }
                    else{
                        adjMatrix[idx][neigh_idx] = 0;
                        adjMatrix[neigh_idx][idx] = 0;
                    }
                }
            }
        }
        std::cout << "Epoch " << i << " completed." << std::endl;
    }
    return adjMatrix;
}




QVector<QVector<double>> SegmentImg::createLaplacianMatrix(QVector<QVector<double>> &adjMatrix, QVector<QVector<double>> &degreeMatrix){
    int n = adjMatrix.size();
    QVector<QVector<double>> laplacianMatrix(n, QVector<double>(n,0));

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            laplacianMatrix[i][j] = degreeMatrix[i][j] - adjMatrix[i][j];
        }
    }

    /*
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            std::cout << i+j << ": " << laplacianMatrix[i][j] << std::endl;
        }
        break;
    }
    */

    return laplacianMatrix;
}


QVector<QVector<double>> SegmentImg::createDegreeMatrix(QVector<QVector<double>> &adjMatrix){
    int n = adjMatrix.size();
    QVector<QVector<double>> degreeMatrix(n, QVector<double>(n, 0));

    //create degree matrix
    for (int i = 0; i < n; i++){
        int degree = 0;
        for (int j = 0; j < n; j++){
            // if (adjMatrix[i][j] > 0){
            //     degree += 1;
            // }
            degree += adjMatrix[i][j];
        }
        degreeMatrix[i][i] = degree; //degree matrix is always diagonal as it just counts # of relationships for a vertice
    }

    return degreeMatrix;
}





//convert matrix type to be compatible with eigen library
Eigen::MatrixXd convertToEigenMat(const QVector<QVector<double>> qtMat){
    //technically I make the matrix square in my adjMat calc so i could just set rows and cols = qtMat.size()
    int rows = qtMat.size();
    int cols = qtMat[0].size();
    Eigen::MatrixXd eigenMat(rows, cols);
    std::cout << "ROWS: " << rows << " | COLS: " << cols << std::endl;

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            eigenMat(i,j) = qtMat[i][j];
        }
    }
    return eigenMat;
}


