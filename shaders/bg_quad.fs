#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uFrame;

void main() {
    vec3 color = texture(uFrame, vUV).rgb;
    FragColor = vec4(color, 1.0);
}
