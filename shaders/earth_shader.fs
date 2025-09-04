#version 330 core

out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D diffuseMap;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float shininess;

void main()
{
    // Use texture color
    vec3 color = texture(diffuseMap, TexCoords).rgb;  

    // Ambient
    vec3 ambient = 0.8 * color;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;

    // Specular (Blinn-Phong)
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = vec3(0.1) * spec; // Adjust specular strength

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}