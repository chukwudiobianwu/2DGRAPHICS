// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "raster.h"

#include <gif.h>
#include <fstream>

#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

//Image height
const int H = 480;

//Camera settings
const double near_plane = 1.5; //AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; //45 degrees
const double aspect_ratio = 1.5;
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

//Object
const std::string data_dir = DATA_DIR;
const std::string mesh_filename(data_dir + "bunny.off");
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)

//Material for the object
const Vector3d obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3d obj_specular_color(0.2, 0.2, 0.2);
const double obj_specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
//Ambient light
const Vector3d ambient_light(0.3, 0.3, 0.3);

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    if (!in.good())
    {
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

void build_uniform(UniformAttributes &uniform)
{
    // TODO: setup uniform

    // TODO: setup camera, compute w, u, v
    const Vector3d w = -camera_gaze.normalized();

    const Vector3d u = camera_top.cross(w).normalized();

    const Vector3d v = w.cross(u);

    // TODO: compute the camera transformation

    Matrix4f slowet;
    Matrix4f don;
    slowet << u(0), v(0), w(0), camera_position(0),
        u(1), v(1), w(1), camera_position(1),
        u(2), v(2), w(2), camera_position(2),
        0, 0, 0, 1;
    // TODO: setup projection matrix
    don << 2 / (((near_plane * tan(field_of_view / 2.0)) * aspect_ratio) - (-((near_plane * tan(field_of_view / 2.0)) * aspect_ratio))), 0, 0, -(((near_plane * tan(field_of_view / 2.0)) * aspect_ratio) + (-((near_plane * tan(field_of_view / 2.0)) * aspect_ratio))) / (((near_plane * tan(field_of_view / 2.0)) * aspect_ratio) - (-((near_plane * tan(field_of_view / 2.0)) * aspect_ratio))),
        0, 2 / ((near_plane * tan(field_of_view / 2.0)) - (-(near_plane * tan(field_of_view / 2.0)))), 0, -((near_plane * tan(field_of_view / 2.0)) + (-(near_plane * tan(field_of_view / 2.0)))) / ((near_plane * tan(field_of_view / 2.0)) - (-(near_plane * tan(field_of_view / 2.0)))),
        0, 0, 2 / (-near_plane - (-far_plane)), -(-near_plane + (-far_plane)) / (-near_plane - (-far_plane)),
        0, 0, 0, 1;

    Matrix4d P;
    if (is_perspective)
    {
        // TODO setup prespective camera
        P << (-near_plane), 0, 0, 0,
            0, (-near_plane), 0, 0,
            0, 0, ((-near_plane) + (-far_plane)), (-(-far_plane) * (-near_plane)),
            0, 0, 1, 0;

        uniform.view = don * P.cast<float>() * (slowet.inverse());
    }
}

void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: fill the shader
        // return va;

        VertexAttributes slo;
        slo.position = uniform.view * va.position;
        return slo;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous)
    {
        // TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;
    // TODO: build the vertex attributes from vertices and facets

    // rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);

for (int i = 0; i < facets.rows(); ++i)
{
    // Extract the vertices of the triangle from the vertices array
    Eigen::Vector3d v1 = vertices.row(facets(i, 0)).transpose();
    Eigen::Vector3d v2 = vertices.row(facets(i, 1)).transpose();
    Eigen::Vector3d v3 = vertices.row(facets(i, 2)).transpose();

    // Add vertex attributes to the vector using a single push_back statement
    vertex_attributes.push_back(VertexAttributes(v1(0), v1(1), v1(2)));
    vertex_attributes.push_back(VertexAttributes(v2(0), v2(1), v2(2)));
    vertex_attributes.push_back(VertexAttributes(v3(0), v3(1), v3(2)));
}

    float aspect_ratio = static_cast<float>(frameBuffer.cols()) / static_cast<float>(frameBuffer.rows());
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    if (aspect_ratio < 1) {
        view(0, 0) = aspect_ratio;
    } else {
        view(1, 1) = 1 / aspect_ratio;
    }

    uniform.view = view;


    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}


Matrix4d compute_rotation(const double alpha)
{
    // Compute the object barycenter
  // Compute the rotation matrix
Matrix4d rotation_matrix;
rotation_matrix << (cos(alpha)), 0, (sin(alpha)), 0,
                   0, 1, 0, 0,
                   -(sin(alpha)), 0, (cos(alpha)), 0,
                   0, 0, 0, 1;
// Apply the identity matrix to the rotation matrix and return the result
Matrix4d identity_matrix;
identity_matrix.setIdentity();
return rotation_matrix * identity_matrix;
}


void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    Matrix4d trafo = compute_rotation(alpha);

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        VertexAttributes slo;
        slo.position = uniform.view * va.position;
        return slo;
        return va;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    //TODO: generate the vertex attributes for the edges and rasterize the lines
    //TODO: use the transformation matrix
for (int i = 0; i < facets.rows(); ++i)
{
    // Get points from triangle
    Vector3i verticesIndex;
    verticesIndex << facets.row(i).transpose();

    // Add edges to vertex_attributes vector
    for (int j = 0; j < 3; ++j)
    {
        Vector3d v1 = vertices.row(verticesIndex(j)).transpose();
        Vector3d v2 = vertices.row(verticesIndex((j + 1) % 3)).transpose();
        vertex_attributes.emplace_back(v1[0], v1[1], v1[2]);
        vertex_attributes.emplace_back(v2[0], v2[1], v2[2]);
    }
}

uniform.view = uniform.view * trafo.cast<float>();

// Clear the frame buffer before rasterizing the lines
frameBuffer.setZero();

rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);

}

void get_shading_program(Program &program)
{
 program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // The only difference lies in the attributes that are sent to the vertex sharer.

        // TODO: transform the position and the normal
        VertexAttributes slo;
        slo.position = uniform.view * va.position;
        slo.normal = va.normal;
        Vector3d vcoller(0, 0, 0);
        for (int i = 0; i < light_positions.size(); i++)
        {
            const Vector3d light_position = light_positions[i];
            const Vector3d light_color = light_colors[i];

            // need to make p and N so we can port over old code
            Vector3d p(slo.position[0], slo.position[1], slo.position[2]);
            Vector3d N(va.normal[0], va.normal[1], va.normal[2]);

            // Diffuse contribution
            const Vector3d diffuse = obj_diffuse_color * std::max(((light_position - p).normalized()).dot(N), 0.0);

            // Specular contribution
            const Vector3d specular = obj_specular_color * std::pow(std::max(N.dot((((light_position - p).normalized()) - p).normalized()), 0.0), obj_specular_exponent);

            // Attenuate lights according to the squared distance to the lights
            vcoller += (diffuse + specular).cwiseProduct(light_color) / (light_position - p).squaredNorm();
        }
        vcoller += ambient_light;

        Vector4f C(vcoller[0], vcoller[1], vcoller[2], 1);
        slo.color = C;
        return slo;
    };
    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: create the correct fragment
        FragmentAttributes slo(va.color[0], va.color[1], va.color[2], uniform.color[3]);
        Vector4f loki(va.position[0], va.position[1], -va.position[2], va.position[3]);
        if (is_perspective)
        {
            loki[2] = va.position[2];
        }
        slo.position = loki; // TRY doing slo.position = va.position
        return slo;
    };


program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous)
    {
        FrameBufferAttributes out = previous;
        // Check if the fragment's depth is less than the previous depth
        if (fa.position(2) < previous.depth)
        {
            out = FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
            out.depth = fa.position[2];
        }

        return out;
    };

}

void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;    // same from simple and wireframe
    build_uniform(uniform);       // same from simple and wireframe
    Program program;              // same from simple and wireframe
    get_shading_program(program); // This is new, will update program

    Eigen::Matrix4d trafo = compute_rotation(alpha); // This is new

    std::vector<VertexAttributes> vertex_attributes;
    // TODO: compute the normals

// Iterate through each triangular facet
for (int i = 0; i < facets.rows(); i++)
{
    // Extract the three vertices of the current facet
    Vector3d vertex_a = vertices.row(facets(i, 0)).transpose();
    Vector3d vertex_b = vertices.row(facets(i, 1)).transpose();
    Vector3d vertex_c = vertices.row(facets(i, 2)).transpose();

    // Create three vertex attributes with the same normal
    VertexAttributes a(vertex_a[0], vertex_a[1], vertex_a[2]);
    a.normal = (vertex_b - vertex_a).cross(vertex_c - vertex_a).normalized().cast<float>();
    VertexAttributes b(vertex_b[0], vertex_b[1], vertex_b[2]);
    b.normal = (vertex_b - vertex_a).cross(vertex_c - vertex_a).normalized().cast<float>();
    VertexAttributes c(vertex_c[0], vertex_c[1], vertex_c[2]);
    c.normal = (vertex_b - vertex_a).cross(vertex_c - vertex_a).normalized().cast<float>();

    // Add the three vertex attributes to the list
    vertex_attributes.emplace_back(a);
    vertex_attributes.emplace_back(b);
    vertex_attributes.emplace_back(c);
}

    float aspect_ratio = float(frameBuffer.cols()) / float(frameBuffer.rows());

    uniform.view *= trafo.cast<float>();
    // TODO: set material colors

    uniform.color = {0, 0, 0, 1};
    frameBuffer.setZero();
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}


void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>& frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);

    Eigen::Matrix4d trafo = compute_rotation(alpha);

    uniform.view *= trafo.cast<float>();

    std::vector<VertexAttributes> vertex_attributes;
    std::vector<Vector3f> Normals(vertices.rows(), Vector3f(0, 0, 0));

    for (int i = 0; i < facets.rows(); ++i)
    {
        Vector3d v1 = vertices.row(facets(i, 0)).transpose();
        Vector3d v2 = vertices.row(facets(i, 1)).transpose();
        Vector3d v3 = vertices.row(facets(i, 2)).transpose();

        Normals[facets(i, 0)] += (v3 - v1).cross(v2 - v1).normalized().cast<float>();
        Normals[facets(i, 1)] += (v3 - v1).cross(v2 - v1).normalized().cast<float>();
        Normals[facets(i, 2)] += (v3 - v1).cross(v2 - v1).normalized().cast<float>();

        VertexAttributes attr1(v1[0], v1[1], v1[2]);
        VertexAttributes attr2(v2[0], v2[1], v2[2]);
        VertexAttributes attr3(v3[0], v3[1], v3[2]);

        attr1.normal = Normals[facets(i, 0)].normalized();
        attr2.normal = Normals[facets(i, 1)].normalized();
        attr3.normal = Normals[facets(i, 2)].normalized();

        vertex_attributes.push_back(attr1);
        vertex_attributes.push_back(attr2);
        vertex_attributes.push_back(attr3);
    }

    uniform.color = {0, 0, 0, 1};
    frameBuffer.setZero();
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}


int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;

    simple_render(frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    //TODO: add the animation
 // Define the GIF filenames and frame delay
const std::vector<std::string> gifNames = {"wireframe_render.gif", "flat_shading.gif", "pv_shading.gif"};
const int delay = 25;
using RenderFunc = std::function<void(float, FrameBuffer&)>;
const std::vector<RenderFunc> renderFuncs = {wireframe_render, flat_shading, pv_shading};
// Iterate over each GIF filename and rendering function
    for (size_t i = 0; i < gifNames.size(); ++i) {
        const std::string& gifName = gifNames[i];
        const RenderFunc& renderFunc = renderFuncs[i];

        // Create the GIF file and iterate over the frames
    const int delay = 25;
    GifWriter g;
    int rows = frameBuffer.rows();
    int cols = frameBuffer.cols();
    GifBegin(&g, gifName.c_str(), rows, cols, delay);

    for (int i = 0; i < 13; ++i) {
        float angle = 1.f + (EIGEN_PI / 12.f) * i;
        frameBuffer.setConstant(FrameBufferAttributes());
        renderFunc(angle, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), rows, cols, delay);
    }

    GifEnd(&g);

    }


    return 0;
}