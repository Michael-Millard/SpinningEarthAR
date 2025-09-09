#ifndef MY_MODEL_HPP
#define MY_MODEL_HPP

#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stb_image.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <my_mesh.hpp>
#include <my_shader.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <functional>

class Model
{
public:
    // Constructor (expects a filepath to a 3D model)
    Model(std::string const& objPath, const std::string& modelName) {
        modelName_ = modelName;
        loadModel(objPath);
        printModelDetails();
    }

    // Draw the model (all its meshes)
    void draw(Shader& shader) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(meshes_.size()); i++) {
            meshes_[i].draw(shader);
        }
    }

    // Draw with a per-mesh transform provider (returns a mesh-space transform for a mesh name)
    void drawWithTransforms(Shader& shader, const std::function<glm::mat4(const std::string&)>& getTransform) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(meshes_.size()); i++) {
            const std::string& name = meshes_[i].meshName_;
            glm::mat4 mm = glm::mat4(1.0f);
            if (getTransform) {
                mm = getTransform(name);
            }
            meshes_[i].draw(shader, mm);
        }
    }

private:
    std::string modelName_;
    std::vector<Mesh> meshes_;
    std::vector<Texture> loadedTextures_;

    // Load a 3D model specified by path
    void loadModel(std::string const& path) {
        // Read file
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        
        // Check for errors
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
            return;
        }

        // Process ASSIMP's root node recursively
        processNode(scene->mRootNode, scene);
    }

    // Processes a node recursively
    void processNode(aiNode* node, const aiScene* scene) {
        // Process each mesh located at current node
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes_.push_back(processMesh(mesh, scene));
        }
        // Recursively process children nodes
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene);
        }
    }

    Mesh processMesh(aiMesh* mesh, const aiScene* scene) {
        // Data to fill
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        std::vector<Texture> textures;

        // Loop through mesh's vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Vertex vertex;
            glm::vec3 vector;

            // Positions
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.position = vector;

            // Normals (if it has)
            if (mesh->HasNormals()) {
                vector.x = mesh->mNormals[i].x;
                vector.y = mesh->mNormals[i].y;
                vector.z = mesh->mNormals[i].z;
                vertex.normal = vector;
            }

            // Texture coords
            if (mesh->mTextureCoords[0]) {
                glm::vec2 vec;
                vec.x = mesh->mTextureCoords[0][i].x;
                vec.y = mesh->mTextureCoords[0][i].y;
                vertex.texCoords = vec;
            } else {
                vertex.texCoords = glm::vec2(0.0f, 0.0f);
            }

            vertices.push_back(vertex);
        }

        // Loop through mesh's faces and retrieve the corresponding vertex indices
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];

            // Retrieve all indices of the face and store them in the indices vector
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }

        // Process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        
        // Diffuse textures
        std::vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "diffuseMap");
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

        // Normal map textures
        std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_NORMALS, "normalMap");
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

        // Bump map textures
        std::vector<Texture> bumpMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "bumpMap");
        textures.insert(textures.end(), bumpMaps.begin(), bumpMaps.end());

        return Mesh(vertices, indices, textures, std::string(mesh->mName.C_Str()));
    }

    // Load materials
    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName) {
        std::vector<Texture> textures;
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++) {
            aiString str;
            mat->GetTexture(type, i, &str);

            // Check if texture already loaded and if so, continue
            bool skip = false;
            for (int j = 0; j < static_cast<int>(loadedTextures_.size()); j++) {
                if (std::strcmp(loadedTextures_[j].path.data(), str.C_Str()) == 0) {
                    textures.push_back(loadedTextures_[j]);
                    skip = true; 
                    break;
                }
            }
            if (!skip) {
                Texture texture;
                texture.id = loadTexture(str.C_Str());
                texture.type = typeName;
                texture.path = str.C_Str();
                textures.push_back(texture);
                loadedTextures_.push_back(texture);
            }
        }
        return textures;
    }

    // Load texture
    unsigned int loadTexture(const char* texturePath) {
        unsigned int textureID;
        glGenTextures(1, &textureID);

        stbi_set_flip_vertically_on_load(false);
        int width, height, numChannels;
        unsigned char* data = stbi_load(texturePath, &width, &height, &numChannels, 0);
        if (data) {
            GLenum format = GL_RGB;
            if (numChannels == 1) {
                format = GL_RED;
            } else if (numChannels == 3) {
                format = GL_RGB;
            } else if (numChannels == 4) {
                format = GL_RGBA;
            }

            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        } else {
            std::cout << "Texture failed to load at path: " << texturePath << std::endl;
        }

        stbi_image_free(data);
        return textureID;
    }

    void printModelDetails() {
        unsigned int totalVertices = 0;
        unsigned int totalTriangles = 0;

        for (const auto& mesh : meshes_) {
            totalVertices += static_cast<unsigned int>(mesh.vertices_.size());
            totalTriangles += static_cast<unsigned int>(mesh.indices_.size()) / 3;
        }

        std::cout << "****************************\n";
        std::cout << "Successfully Loaded Model: " << modelName_ << "\n";
        std::cout << "Model contains " << meshes_.size() << " mesh(es).\n";
        std::cout << "Total vertices: " << totalVertices << "\n";
        std::cout << "Total triangles: " << totalTriangles << "\n";
        std::cout << "****************************\n\n";
    }
};

#endif // MY_MODEL_HPP
