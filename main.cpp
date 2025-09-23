#include <igl/readOFF.h>
#include <igl/writeOFF.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/cotmatrix.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/adjacency_list.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>
#include <GLFW/glfw3.h>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <unordered_set>
#include <iostream>

bool selectHandles = true;
bool allAnchors = false;

enum meshes{BUNNY, COW, LION};
meshes mesh = BUNNY;

//Solver for retaining certain things like A, b, handles, anchors
class LaplacianDeformer {

public:
    std::vector<std::vector<int>> adjacencyList;
    Eigen::SparseMatrix<double> L;              
    Eigen::SparseMatrix<double> AtA;            
    Eigen::SparseMatrix<double> At;
    Eigen::SparseMatrix<double> A;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver; 
    std::unordered_set<int> handles;
    std::unordered_set<int> anchors;
    std::vector<Eigen::Matrix3d> R;
    std::vector<Eigen::Triplet<double>> A_triplets; 
    Eigen::MatrixXd b;
    int n, m;//n = vertices, m = constraints
    int maxIterations = 5;

    LaplacianDeformer() : n(0), m(0) {}

    void initialize(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
        igl::cotmatrix(V, F, L);
        n = V.rows();
        m = 0;
        igl::adjacency_list(F, adjacencyList);
        update_constraints();
    }


    void update_constraints() {
        m = handles.size() + anchors.size();
        A_triplets.clear();
        A_triplets.reserve(L.nonZeros() + m); //Pre-allocate

        //Copy Laplacian entries
        for (int k = 0; k < L.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
                A_triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));

        //Add constraint rows (0s and 1s)
        int row = n;
        for (int i : handles)
            A_triplets.push_back(Eigen::Triplet<double>(row++, i, 1.0));
        for (int i : anchors)
            A_triplets.push_back(Eigen::Triplet<double>(row++, i, 1.0));

        //Build A and AtA
        Eigen::SparseMatrix<double> A(n + m, n);
        A.setFromTriplets(A_triplets.begin(), A_triplets.end());
        this->A = A;
        At = A.transpose();
        AtA = At * A;

        // Factorize
        solver.compute(AtA);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Cholesky decomposition failed");
    }
    
    void setupb(const Eigen::MatrixXd& V, const Eigen::MatrixXd& Vprime) {
        Eigen::MatrixXd b(n + m, 3);
        b.topRows(n) = L * V;
        int row = n;
        for (int i : handles)
            b.row(row++) = Vprime.row(i);
        for (int i : anchors)
            b.row(row++) = Vprime.row(i);
        this->b = b;
    }

    Eigen::MatrixXd deform() {
        //// Build b = [delta; V'_constraints]
        //Eigen::MatrixXd b(n + m, 3);
        //b.topRows(n) = L * V;
        //std::cout << b.row(0).rows() << " " << b.row(0).cols() << "\n";

        ///*if (!R.empty()) {
        //    for (int i = 0; i < n; i++) {
        //        b.row(i) = R[i] * b.row(i).transpose();
        //    }
        //}*/

        //int row = n;
        //for (int i : handles)
        //    b.row(row++) = Vprime.row(i);
        //for (int i : anchors)
        //    b.row(row++) = Vprime.row(i);

        // Compute A^T b
        /*Eigen::SparseMatrix<double> A(n + m, n);
        A.setFromTriplets(A_triplets.begin(), A_triplets.end());*/ 
        Eigen::MatrixXd b_temp = b;

        if (!R.empty()) {
            for (int i = 0; i < R.size(); i++) {
                b_temp.row(i) = R[i] * b_temp.row(i).transpose();
            }
        }

        Eigen::MatrixXd Atb = At * b_temp;

        //Solve
        Eigen::MatrixXd deformed = solver.solve(Atb);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Solve failed");

        return deformed;
    }

    /*Eigen::MatrixXd iterator(const Eigen::MatrixXd& V, const Eigen::MatrixXd& Vprime) {
        Eigen::MatrixXd finalVPrime;
        
    }*/

    void computeRotation(const Eigen::MatrixXd& vOriginal,const Eigen::MatrixXd& vTransformed) {
        int n = vOriginal.rows();
       
        std::vector<Eigen::Matrix3d> R(vOriginal.rows());

        for (int i = 0; i < vOriginal.rows(); i++) {
            if (anchors.find(i) != anchors.end()) {
                R[i] = Eigen::Matrix3d::Identity();
                continue;
            }
            const std::vector<int>& adjacentVertices = adjacencyList[i];
            Eigen::Matrix3d S = Eigen::Matrix3d::Zero();

            for (int j = 0; j < adjacentVertices.size(); j++) {
                int adjacentVertexIndex = adjacentVertices[j];
                Eigen::Vector3d delta_v = vOriginal.row(i) - vOriginal.row(adjacentVertexIndex);
                Eigen::Vector3d delta_vPrime = vTransformed.row(i) - vTransformed.row(adjacentVertexIndex);
                double w_ij = std::abs(L.coeff(i, adjacentVertexIndex));
                S += w_ij * delta_v * delta_vPrime.transpose();
            }

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            R[i] = (V * U.transpose());

            if (R[i].determinant() < 0) {
                V.col(2) *= -1;
                R[i] = V * U.transpose();
            }
        }

        this->R = R;
    }

    Eigen::MatrixXd iterate(const Eigen::MatrixXd& vOriginal) {
        //Eigen::MatrixXd current = vTransformed;
        for (int i = 0; i < maxIterations; i++) {
            computeRotation(vOriginal, deform());
            
        }
        return deform();
    }

};

bool saveMesh(const std::string& filename,Eigen::MatrixXd V, Eigen::MatrixXi F) {
    return igl::writeOFF(filename, V, F);
}

//void initialize(Eigen::MatrixXd V, Eigen::MatrixXi F) {
//
//    std::string currentMesh;
//
//    if (mesh == BUNNY) {
//        currentMesh = "../../../bunny.off";
//    }
//    else if (mesh == COW) {
//        currentMesh = "../../../cow.off";
//    }
//    else {
//        currentMesh = "../../../lion.off";
//    }
//
//    if (!igl::readOFF(currentMesh, V, F)) {
//        std::cerr << "Failed to load mesh!" << std::endl;
//
//    }
//}

//void initialize(Eigen::MatrixXd& V, Eigen::MatrixXd& V_transformed, Eigen::MatrixXd& V_original, Eigen::MatrixXi F, Eigen::MatrixXd C, Eigen::MatrixXd defaultC, LaplacianDeformer& deformer, igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo) {
//    std::string currentMesh;
//    
//    if (mesh == BUNNY) {
//        currentMesh = "../../../bunny.off";
//    }
//    else if (mesh == COW) {
//        currentMesh = "../../../cow.off";
//    }
//    else {
//        currentMesh = "../../../lion.off";
//    }
//
//    if (!igl::readOFF(currentMesh, V, F)) {
//        std::cerr << "Failed to load mesh!" << std::endl;
//
//    }
//
//    V_transformed = V;
//    V_original = V;
//
//    C = Eigen::MatrixXd::Constant(V.rows(), 3, 0.7);
//    defaultC = C;
//
//    LaplacianDeformer* deformer = new LaplacianDeformer();
//
//    deformer.initialize(V, F);
//
//}

int main()
{
    
    Eigen::MatrixXd V, C, V_transformed;
    Eigen::MatrixXi F;

    std::string currentMesh;

    bool meshSelected = false;

    while (!meshSelected) {

        int value = 0;
        std::cout << "Please press 1 for a bunny mesh, 2 for a cow mesh, and 3 for a lion mesh." << std::endl;
        std::cin >> value;

        if (value -1 == BUNNY) {
            mesh = BUNNY;
            currentMesh = "../../../bunny.off";
            meshSelected = true;
        }
        else if (value - 1 == COW) {
            mesh = COW;
            currentMesh = "../../../cow.off";
            meshSelected = true;
        }
        else {
            mesh = LION;
            currentMesh = "../../../lion.off";
            meshSelected = true;
        }

    }

    
    if (!igl::readOFF(currentMesh, V, F)) {
        std::cerr << "Failed to load mesh!" << std::endl;
        return 1;
    }

    V_transformed = V;
    Eigen::MatrixXd V_original = V;

    
    C = Eigen::MatrixXd::Constant(V.rows(), 3, 0.7);
    Eigen::MatrixXd defaultC = C;

    
    LaplacianDeformer deformer;
    deformer.initialize(V, F);

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);

    igl::opengl::glfw::imgui::ImGuizmoWidget gizmo;
    imgui_plugin.widgets.push_back(&gizmo);

    gizmo.T.block(0, 3, 3, 1) = 0.5 * (V.colwise().maxCoeff() + V.colwise().minCoeff()).transpose().cast<float>();
    const Eigen::Matrix4f T0 = gizmo.T;

    
    gizmo.callback = [&](const Eigen::Matrix4f& T) {
        if (deformer.handles.empty() || deformer.anchors.empty())
            return;
        const Eigen::Matrix4d TT = (T * T0.inverse()).cast<double>().transpose();
        //V_transformed = V_original;

        for (int vid : deformer.handles)
            V_transformed.row(vid) = (V_original.row(vid).homogeneous() * TT).hnormalized();

        deformer.setupb(V, V_transformed);
        V_transformed = deformer.iterate(V);
        
        //deformer.R.clear();
        viewer.data().set_vertices(V_transformed);
        viewer.data().compute_normals();
        V = V_transformed;
    };

    
    viewer.callback_key_pressed = [&](decltype(viewer)&, unsigned int key, int mod) {
        
        
        switch (key) {
            case 'C': case 'c': selectHandles = !selectHandles; return true;
            case ' ': gizmo.visible = !gizmo.visible; return true;
            case 'W': case 'w': gizmo.operation = ImGuizmo::TRANSLATE; return true;
            case 'E': case 'e': gizmo.operation = ImGuizmo::ROTATE; return true;
            case 'R': case 'r': gizmo.operation = ImGuizmo::SCALE; return true;
            case 'H': case 'h': {
                if (allAnchors) {
                    deformer.anchors.clear();
                    deformer.handles.clear();
                    C = defaultC;
                    allAnchors = false;
                }
                else {
                    for (int i = 0; i < V.rows(); i++) {
                        deformer.anchors.insert(i);
                        C.row(i) << 0.0, 0.0, 0.8;
                    }
                    deformer.handles.clear();
                    allAnchors = true;
                }
                deformer.update_constraints();
                viewer.data().set_colors(C);
                return true;
            }
            case 'I': case 'i': {
                deformer.maxIterations++;
                std::cout << "maxIterations increased to " << deformer.maxIterations << std::endl;
                return true;
            }
            case 'K': case 'k':{
                deformer.maxIterations = std::max(1, --deformer.maxIterations);
                std::cout << "maxIterations decreased to " << deformer.maxIterations << std::endl;
                return true;
            }
            case 'N': case 'n': {
                if (mesh == BUNNY) {
                    if (igl::writeOFF("../../../newBunny.off", V, F))
                        std::cout << "Mesh has been successfully saved." << std::endl;
                    else
                        std::cout << "Error: Failed to write newBunny.off" << std::endl;
                    return true;
                }
                else if (mesh == COW) {
                    if (igl::writeOFF("../../../newCow.off", V, F))
                        std::cout << "Mesh has been successfully saved." << std::endl;
                    return true;
                }
                else {
                    if (igl::writeOFF("../../../newLion.off", V, F))
                        std::cout << "Mesh has been successfully saved." << std::endl;
                    return true;
                }
                
            }

        }

        
        return false;
    };

    // Mouse callback
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool {
        int fid;
        Eigen::Vector3f bc;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, fid, bc)) {
            int dominant_vertex_index = (bc[1] > bc[0]) ? (bc[2] > bc[1] ? 2 : 1) : (bc[2] > bc[0] ? 2 : 0);
            int vid = F(fid, dominant_vertex_index);

            std::cout << "Selected Vertex: " << vid << std::endl;

            if (selectHandles) {
                if (deformer.handles.find(vid) != deformer.handles.end()) {
                    deformer.handles.erase(vid);
                    C.row(vid) << 0.7, 0.7, 0.7;
                }
                else {
                    if (deformer.anchors.find(vid) != deformer.anchors.end())
                        deformer.anchors.erase(vid);
                    deformer.handles.insert(vid);
                    C.row(vid) << 0.8, 0.0, 0.0;
                }
            }
            else {
                if (deformer.anchors.find(vid) != deformer.anchors.end()) {
                    deformer.anchors.erase(vid);
                    C.row(vid) << 0.7, 0.7, 0.7;
                }
                else {
                    if(deformer.handles.find(vid) != deformer.handles.end())
                        deformer.handles.erase(vid);
                    deformer.anchors.insert(vid);
                    C.row(vid) << 0.0, 0.0, 0.8;
                }
            }

            //Update deformer when constraints change
            deformer.update_constraints();
            viewer.data().set_colors(C);
            return true;
        }
        return false;
    };

    //Launch viewer
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.launch();
    return 0;
}