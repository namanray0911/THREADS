import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import math
from scipy.interpolate import splprep, splev
import time

# ======================== SHAPE GENERATORS ========================
def circle_shape(r, cx=0, cy=0, segments=20):
    """Generate a circle with given radius and center"""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

def polygon_shape(n_sides, r, cx=0, cy=0):
    """Generate a regular polygon with n sides"""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

def ellipse_shape(r, aspect_ratio, cx=0, cy=0, segments=30):
    """Generate an ellipse with given aspect ratio"""
    angles = np.linspace(0, 2 * np.pi, segments)
    return [(cx + r * np.cos(a), cy + r * aspect_ratio * np.sin(a)) for a in angles]

def star_shape(n_points, r, cx=0, cy=0):
    """Generate a star shape with n points"""
    angles = np.linspace(0, 2 * np.pi, n_points * 2, endpoint=False)
    radius = [r if i % 2 == 0 else r * 0.5 for i in range(len(angles))]
    return [(cx + radius[i] * np.cos(angles[i]), cy + radius[i] * np.sin(angles[i])) for i in range(len(angles))]

def superellipse_shape(r, n=4, cx=0, cy=0, segments=50):
    """Generate a superellipse (squircle) shape"""
    angles = np.linspace(0, 2 * np.pi, segments)
    points = []
    for a in angles:
        x = r * np.sign(np.cos(a)) * abs(np.cos(a))**(2/n)
        y = r * np.sign(np.sin(a)) * abs(np.sin(a))**(2/n)
        points.append((cx + x, cy + y))
    return points

def random_shape(r, cx=0, cy=0, segments=30):
    """Generate a random organic shape"""
    base_circle = circle_shape(r, cx, cy, segments)
    base_circle = np.array(base_circle)
    
    # Add random perturbations
    perturbations = np.random.normal(0, 0.2*r, (segments, 2))
    smoothed_perturbations = np.zeros_like(perturbations)
    
    # Smooth the perturbations
    for i in range(2):
        tck, _ = splprep([perturbations[:, i]], s=segments)
        smoothed_perturbations[:, i] = splev(np.linspace(0, 1, segments), tck)[0]
    
    random_shape = base_circle + smoothed_perturbations
    return [(p[0], p[1]) for p in random_shape]

# ======================== THREAD GENERATION ========================
def generate_threads(Lx, Ly, min_r, max_r, num_threads, materials, packing='grid'):
    threads = []
    placed = 0
    max_attempts = num_threads * 1000
    
    if packing == 'grid':
        grid_size = int(np.sqrt(num_threads) * 1.2)
        x_grid = np.linspace(max_r, Lx - max_r, grid_size)
        y_grid = np.linspace(max_r, Ly - max_r, grid_size)
        
        for x in x_grid:
            for y in y_grid:
                if placed >= num_threads:
                    break
                
                jitter_x = np.random.uniform(-max_r/2, max_r/2)
                jitter_y = np.random.uniform(-max_r/2, max_r/2)
                x_jittered = x + jitter_x
                y_jittered = y + jitter_y
                
                if x_jittered < max_r or x_jittered > Lx - max_r:
                    continue
                if y_jittered < max_r or y_jittered > Ly - max_r:
                    continue
                
                r = np.random.uniform(min_r, max_r)
                material = random.choice(materials)
                waviness = np.random.uniform(0.05 * r, 0.2 * r)
                roughness = np.random.uniform(0.03, 0.1)
                max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
                
                if not check_collision(x_jittered, y_jittered, r, threads):
                    threads.append((x_jittered, y_jittered, r, material, waviness, roughness, max_length))
                    placed += 1
    elif packing == 'hexagonal':
        spacing = max_r * 2 * 1.05
        rows = int((Ly - 2*max_r) / (spacing * np.sqrt(3)/2)) + 1
        cols = int((Lx - 2*max_r) / spacing) + 1
        
        for row in range(rows):
            for col in range(cols):
                if placed >= num_threads:
                    break
                
                x = max_r + col * spacing
                if row % 2 == 1:
                    x += spacing / 2
                
                y = max_r + row * spacing * np.sqrt(3)/2
                
                if x > Lx - max_r or y > Ly - max_r:
                    continue
                
                r = np.random.uniform(min_r, max_r)
                material = random.choice(materials)
                waviness = np.random.uniform(0.05 * r, 0.2 * r)
                roughness = np.random.uniform(0.03, 0.1)
                max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
                
                if not check_collision(x, y, r, threads):
                    threads.append((x, y, r, material, waviness, roughness, max_length))
                    placed += 1
    else:  # random
        attempts = 0
        while placed < num_threads and attempts < max_attempts:
            x = np.random.uniform(max_r, Lx - max_r)
            y = np.random.uniform(max_r, Ly - max_r)
            r = np.random.uniform(min_r, max_r)
            material = random.choice(materials)
            waviness = np.random.uniform(0.05 * r, 0.2 * r)
            roughness = np.random.uniform(0.03, 0.1)
            max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
            
            if not check_collision(x, y, r, threads):
                threads.append((x, y, r, material, waviness, roughness, max_length))
                placed += 1
            attempts += 1
    
    placement_ratio = placed / num_threads * 100
    st.info(f"Successfully placed {placed}/{num_threads} threads ({placement_ratio:.1f}%) using {packing} packing")
    
    if placement_ratio < 90:
        st.warning("Low placement percentage. Consider increasing domain size or decreasing thread count.")
    
    return threads

def check_collision(x, y, r, existing_threads, safety_margin=1.05):
    if not existing_threads:
        return False
    
    existing_pos = np.array([[t[0], t[1]] for t in existing_threads])
    existing_rad = np.array([t[2] * safety_margin for t in existing_threads])
    
    distances = np.sqrt((existing_pos[:, 0] - x)**2 + (existing_pos[:, 1] - y)**2)
    return np.any(distances < (r * safety_margin + existing_rad))

# ======================== 3D THREAD RENDERING ========================
def generate_thread_layers(x, y, r, waviness, roughness, height, segments, shape_type, shape_params):
    layers = []
    z_values = np.linspace(0, height, segments + 1)
    
    if shape_type == "circle":
        base_shape = circle_shape(r, 0, 0)
    elif shape_type == "polygon":
        base_shape = polygon_shape(shape_params['sides'], r, 0, 0)
    elif shape_type == "ellipse":
        base_shape = ellipse_shape(r, shape_params['aspect_ratio'], 0, 0)
    elif shape_type == "star":
        base_shape = star_shape(shape_params['points'], r, 0, 0)
    elif shape_type == "superellipse":
        base_shape = superellipse_shape(r, shape_params['n'], 0, 0)
    elif shape_type == "random":
        base_shape = random_shape(r, 0, 0)
    else:
        base_shape = circle_shape(r, 0, 0)
    
    base_shape = np.array(base_shape)
    
    twist_factor = shape_params.get('twist', 0)
    
    for i, z in enumerate(z_values):
        r_mod = r * (1 + roughness * np.sin(2 * np.pi * z / height))
        cx = x + waviness * np.sin(2 * np.pi * z / height)
        cy = y + waviness * np.cos(2 * np.pi * z / height)
        scaled_shape = base_shape * (r_mod / r)
        
        if twist_factor != 0:
            angle = 2 * np.pi * z * twist_factor
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
            scaled_shape = np.dot(scaled_shape, rot_matrix)
        
        translated_shape = scaled_shape + np.array([cx, cy])
        layers.append([(p[0], p[1], z) for p in translated_shape])
    
    return layers

def draw_threads(ax, threads, Lz, shape_type, shape_params, segments=20, color_by='material'):
    if color_by == 'material':
        materials = list(set([t[3] for t in threads]))
        material_colors = plt.cm.tab20(np.linspace(0, 1, len(materials)))
        color_map = {mat: color for mat, color in zip(materials, material_colors)}
    elif color_by == 'radius':
        radii = [t[2] for t in threads]
        norm = plt.Normalize(min(radii), max(radii))
        cmap = plt.cm.viridis
    elif color_by == 'height':
        heights = [t[6] for t in threads]
        norm = plt.Normalize(min(heights), max(heights))
        cmap = plt.cm.plasma
    
    for idx, (x, y, r, material, waviness, roughness, max_length) in enumerate(threads):
        height = Lz * max_length
        layers = generate_thread_layers(x, y, r, waviness, roughness, height, segments, shape_type, shape_params)
        
        if color_by == 'material':
            color = color_map[material]
        elif color_by == 'radius':
            color = cmap(norm(r))
        elif color_by == 'height':
            color = cmap(norm(max_length))
        
        for i in range(len(layers) - 1):
            verts = []
            n = len(layers[i])
            for j in range(n):
                k = (j + 1) % n
                quad = [layers[i][j], layers[i][k], layers[i+1][k], layers[i+1][j]]
                verts.append(quad)
            
            poly = Poly3DCollection(verts, alpha=0.8)
            poly.set_facecolor(color)
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect([Lx, Ly, Lz])
    
    if color_by in ['radius', 'height']:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label='Radius' if color_by == 'radius' else 'Height fraction')

# ======================== STREAMLIT UI ========================
st.set_page_config(layout="wide", page_title="Composite Thread Simulation")
st.title("Advanced Composite Thread Simulation")

with st.sidebar:
    st.header("Simulation Parameters")
    
    shape_type = st.selectbox(
        "Thread Cross-Section Shape",
        ["circle", "polygon", "ellipse", "star", "superellipse", "random"],
        index=0
    )
    
    shape_params = {}
    if shape_type == "polygon":
        shape_params['sides'] = st.slider("Number of sides", 3, 12, 6)
    elif shape_type == "ellipse":
        shape_params['aspect_ratio'] = st.slider("Aspect ratio", 0.1, 5.0, 1.5)
    elif shape_type == "star":
        shape_params['points'] = st.slider("Number of points", 5, 20, 5)
    elif shape_type == "superellipse":
        shape_params['n'] = st.slider("Exponent (2=ellipse, 4=rounded square)", 2.0, 8.0, 4.0)
    elif shape_type == "random":
        shape_params['twist'] = st.slider("Twist factor", 0.0, 2.0, 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        Lx = st.number_input("Domain Length (X)", min_value=1.0, max_value=100.0, value=10.0)
        Ly = st.number_input("Domain Width (Y)", min_value=1.0, max_value=100.0, value=10.0)
    with col2:
        Lz = st.number_input("Domain Height (Z)", min_value=1.0, max_value=100.0, value=5.0)
        num_threads = st.number_input("Number of Threads", min_value=1, max_value=1000, value=100)
    
    col3, col4 = st.columns(2)
    with col3:
        min_d = st.number_input("Minimum Diameter", min_value=0.01, max_value=5.0, value=0.2)
    with col4:
        max_d = st.number_input("Maximum Diameter", min_value=0.01, max_value=5.0, value=0.5)
    
    packing = st.selectbox(
        "Packing Algorithm",
        ["grid", "hexagonal", "random"],
        index=1
    )
    
    materials = st.multiselect(
        "Materials",
        ["carbon", "glass", "kevlar", "steel", "nylon", "copper"],
        default=["carbon", "glass", "kevlar"]
    )
    
    color_by = st.radio(
        "Color Threads By",
        ["material", "radius", "height"],
        index=0
    )
    
    if st.button("Run Simulation"):
        st.session_state.run_simulation = True

if 'run_simulation' in st.session_state and st.session_state.run_simulation:
    with st.spinner('Generating threads...'):
        threads = generate_threads(Lx, Ly, min_d/2, max_d/2, num_threads, materials, packing)
    
    st.header("3D Visualization")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    with st.spinner('Rendering 3D view...'):
        draw_threads(ax, threads, Lz, shape_type, shape_params, color_by=color_by)
        ax.set_title(f"{len(threads)} {shape_type} threads ({packing} packing)")
        st.pyplot(fig)
    
    st.header("Statistics")
    radii = [t[2] for t in threads]
    heights = [t[6] for t in threads]
    material_counts = {mat: [t[3] for t in threads].count(mat) for mat in materials}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Threads", len(threads))
        st.metric("Average Radius", f"{np.mean(radii):.3f}")
    with col2:
        st.metric("Min Radius", f"{min(radii):.3f}")
        st.metric("Max Radius", f"{max(radii):.3f}")
    with col3:
        st.metric("Average Height", f"{np.mean(heights):.3f}")
        st.metric("Placement Ratio", f"{len(threads)/num_threads*100:.1f}%")
    
    st.subheader("Material Distribution")
    fig2, ax2 = plt.subplots()
    ax2.pie(material_counts.values(), labels=material_counts.keys(), autopct='%1.1f%%')
    st.pyplot(fig2)
    
    st.subheader("Size Distributions")
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    ax3.hist(radii, bins=20, edgecolor='black')
    ax3.set_title('Thread Radius Distribution')
    ax3.set_xlabel('Radius')
    ax4.hist(heights, bins=20, edgecolor='black')
    ax4.set_title('Thread Height Distribution')
    ax4.set_xlabel('Height Fraction')
    st.pyplot(fig3)
    
    st.session_state.run_simulation = False
