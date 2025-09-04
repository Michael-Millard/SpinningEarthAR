#ifndef MY_MESH_H
#define MY_MESH_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <my_shader.h>

#include <string>
#include <vector>

struct Vertex 
{
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct Texture 
{
    unsigned int id;
    std::string type;
    std::string path;
};

class Mesh
{
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    std::string meshName;

    // Init the mesh
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, const std::vector<Texture>& textures, const std::string meshName)
    {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;
        this->meshName = meshName;
        setupMesh();
    }

    // Draw the mesh with identity per-mesh transform
    void draw(Shader& shader)
    {
        // Default mesh transform = identity
        shader.setMat4("meshModel", glm::mat4(1.0f));
        // If multiple textures for this mesh, loop through
        for (unsigned int i = 0; i < static_cast<unsigned int>(textures.size()); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            std::string textureName = textures[i].type;
            shader.setInt(textureName.c_str(), i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glActiveTexture(GL_TEXTURE0);
    }

    // Draw the mesh with a supplied per-mesh transform
    void draw(Shader& shader, const glm::mat4& meshModel)
    {
        shader.setMat4("meshModel", meshModel);
        // If multiple textures for this mesh, loop through
        for (unsigned int i = 0; i < static_cast<unsigned int>(textures.size()); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            std::string textureName = textures[i].type;
            shader.setInt(textureName.c_str(), i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glActiveTexture(GL_TEXTURE0);
    }

private:
    unsigned int VAO, VBO, EBO;

    // Setup
    void setupMesh()
    {
        // Create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        // Bind VAO
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        // EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Vertex positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

        // Vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));

        // Vertex texture coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

        glBindVertexArray(0);
    }
};
#endif
