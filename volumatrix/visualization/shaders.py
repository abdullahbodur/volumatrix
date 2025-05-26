"""
Shader management for Volumatrix visualization.
"""

from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
from typing import Optional, Tuple


class ShaderProgram:
  """Manages OpenGL shader programs for Volumatrix visualization."""

  # Default vertex shader
  DEFAULT_VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoord;
    
    out vec3 FragPos;
    out vec3 Normal;
    out vec2 TexCoord;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        TexCoord = aTexCoord;
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    """

  # Default fragment shader
  DEFAULT_FRAGMENT_SHADER = """
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoord;
    
    out vec4 FragColor;
    
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;
    uniform vec3 objectColor;
    
    void main()
    {
        // ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * lightColor;
        
        // diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        // specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;
        
        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
    """

  # Grid vertex shader
  GRID_VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 view;
    uniform mat4 projection;
    
    void main()
    {
        gl_Position = projection * view * vec4(aPos, 1.0);
    }
    """

  # Grid fragment shader
  GRID_FRAGMENT_SHADER = """
    #version 330 core
    out vec4 FragColor;
    
    uniform vec3 gridColor;
    uniform float gridAlpha;
    
    void main()
    {
        FragColor = vec4(gridColor, gridAlpha);
    }
    """

  def __init__(self, vertex_shader: Optional[str] = None, fragment_shader: Optional[str] = None):
    """Initialize shader program with optional custom shaders."""
    self.vertex_shader = vertex_shader or self.DEFAULT_VERTEX_SHADER
    self.fragment_shader = fragment_shader or self.DEFAULT_FRAGMENT_SHADER
    self.program = self._compile_shaders()

    # Initialize grid shader
    self.grid_program = self._compile_grid_shaders()

  def _compile_shaders(self) -> int:
    """Compile vertex and fragment shaders into a program."""
    try:
      # Compile vertex shader
      vertex = shaders.compileShader(self.vertex_shader, GL_VERTEX_SHADER)

      # Compile fragment shader
      fragment = shaders.compileShader(self.fragment_shader, GL_FRAGMENT_SHADER)

      # Create and link program
      program = shaders.compileProgram(vertex, fragment)

      # Validate program
      glValidateProgram(program)
      if glGetProgramiv(program, GL_VALIDATE_STATUS) != GL_TRUE:
        error = glGetProgramInfoLog(program)
        raise RuntimeError(f"Shader program validation failed: {error}")

      return program

    except Exception as e:
      # Clean up any resources in case of error
      if 'vertex' in locals():
        glDeleteShader(vertex)
      if 'fragment' in locals():
        glDeleteShader(fragment)
      if 'program' in locals():
        glDeleteProgram(program)
      raise RuntimeError(f"Failed to compile shaders: {str(e)}")

  def _compile_grid_shaders(self) -> int:
    """Compile grid shaders into a program."""
    try:
      vertex = shaders.compileShader(self.GRID_VERTEX_SHADER, GL_VERTEX_SHADER)
      fragment = shaders.compileShader(
        self.GRID_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
      program = shaders.compileProgram(vertex, fragment)

      glValidateProgram(program)
      if glGetProgramiv(program, GL_VALIDATE_STATUS) != GL_TRUE:
        error = glGetProgramInfoLog(program)
        raise RuntimeError(f"Grid shader program validation failed: {error}")

      return program

    except Exception as e:
      if 'vertex' in locals():
        glDeleteShader(vertex)
      if 'fragment' in locals():
        glDeleteShader(fragment)
      if 'program' in locals():
        glDeleteProgram(program)
      raise RuntimeError(f"Failed to compile grid shaders: {str(e)}")

  def use(self):
    """Use this shader program."""
    glUseProgram(self.program)

  def use_grid(self):
    """Use the grid shader program."""
    glUseProgram(self.grid_program)

  def set_uniform_matrix4fv(self, name: str, value: np.ndarray):
    """Set a 4x4 matrix uniform."""
    location = glGetUniformLocation(self.program, name)
    if location == -1:
      raise ValueError(f"Uniform '{name}' not found in shader program")
    glUniformMatrix4fv(location, 1, GL_FALSE, value)

  def set_uniform_3fv(self, name: str, value: np.ndarray):
    """Set a 3D vector uniform."""
    location = glGetUniformLocation(self.program, name)
    if location == -1:
      raise ValueError(f"Uniform '{name}' not found in shader program")
    glUniform3fv(location, 1, value)

  def set_uniform_1f(self, name: str, value: float):
    """Set a float uniform."""
    location = glGetUniformLocation(self.program, name)
    if location == -1:
      raise ValueError(f"Uniform '{name}' not found in shader program")
    glUniform1f(location, value)

  def set_grid_uniforms(self, color: np.ndarray, alpha: float):
    """Set uniforms for the grid shader."""
    glUseProgram(self.grid_program)
    color_loc = glGetUniformLocation(self.grid_program, "gridColor")
    alpha_loc = glGetUniformLocation(self.grid_program, "gridAlpha")
    glUniform3fv(color_loc, 1, color)
    glUniform1f(alpha_loc, alpha)

  def set_grid_matrix4fv(self, name: str, value: np.ndarray):
    """Set a 4x4 matrix uniform for the grid shader."""
    location = glGetUniformLocation(self.grid_program, name)
    if location == -1:
      raise ValueError(f"Uniform '{name}' not found in grid shader program")
    glUniformMatrix4fv(location, 1, GL_FALSE, value)

  def cleanup(self):
    """Clean up shader resources."""
    if hasattr(self, 'program'):
      glDeleteProgram(self.program)
    if hasattr(self, 'grid_program'):
      glDeleteProgram(self.grid_program)
