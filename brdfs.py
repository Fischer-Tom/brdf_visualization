import numpy as np
from numpy.linalg import norm


def lambertian(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d, parameters):
    return parameters['rho_d'] / np.pi

def phong(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters):
    normal = np.array([0.,0.,1.])
    incoming = np.array((np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)))
    outgoing = np.array((np.sin(theta_o)*np.cos(phi_o), np.sin(theta_o)*np.sin(phi_o), np.cos(theta_o)))
    reflection_vector = 2*(np.dot(incoming,normal))*normal-incoming
    reflection_vector = reflection_vector / np.linalg.norm(reflection_vector)
    cos = np.dot(outgoing, reflection_vector).round(15)
    specular = parameters['rho_s'] * (np.power(cos,parameters['exp']))
    diffuse = lambertian(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters)

    return specular.round(15) + diffuse


def torrance_sparrow(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters):
    m = parameters['m']
    divisor = 4*np.pi * np.cos(theta_i)

    D_fac = 1/(m*m*(np.cos(theta_h)**4))
    D_exp = (np.cos(theta_h)*np.cos(theta_h)-1)/(m*m*np.cos(theta_h)*np.cos(theta_h))
    D = D_fac * np.exp(D_exp)
    F = 1
    G = 1
    diffuse = lambertian(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters)

    ks = parameters['rho_s']/divisor
    ks *= D*F*G

    return ks + diffuse

def cook_torrance(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters):
    m = parameters['m']
    divisor = np.pi*np.cos(theta_o)* np.cos(theta_i)

    D_fac = 1/(m*m*(np.cos(theta_h)**4))
    D_exp = (np.cos(theta_h)*np.cos(theta_h)-1)/(m*m*np.cos(theta_h)*np.cos(theta_h))
    D = D_fac * np.exp(D_exp)
    F = 0.5
    G = 1
    diffuse = lambertian(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters)

    ks = parameters['rho_s']/divisor
    ks *= D*F*G

    return ks + diffuse

def ward(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters):
    sigma_x = parameters['sigma_x']
    sigma_y = parameters['sigma_y']


    divisor = 4*np.pi*sigma_x*sigma_y*np.sqrt(np.cos(theta_i)*np.cos(theta_o))
    exponent = np.tan(theta_h)*np.tan(theta_h)*(np.cos(phi_h)*np.cos(phi_h)/(sigma_x*sigma_x) + np.sin(phi_h)*np.sin(phi_h)/(sigma_y*sigma_y))

    specular = parameters['rho_s']/divisor * np.exp(-exponent)

    diffuse = lambertian(theta_i,phi_i,theta_o,phi_o,theta_h,phi_h,theta_d,phi_d,parameters)

    return specular + diffuse

def theta(v,w):
    return np.arccos(v.dot(w)/(norm(v)*norm(w)))

brdf_map = {"Lambertian":lambertian,
            "Phong": phong,
            "Torrance-Sparrow": torrance_sparrow,
            "Cook-Torrance": cook_torrance,
            "Ward": ward
            }