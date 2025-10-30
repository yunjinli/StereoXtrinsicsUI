import io
import numpy as np
import cv2
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import yaml
# from tkinter import ttk

from pyopengltk import OpenGLFrame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18

import argparse
import sys
glutInit()

class GLPanel(OpenGLFrame):
    def initgl(self):
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)
        self.ang_x, self.ang_y, self.dist = 20.0, 30.0, 4.0
        self._last = None
        # Mouse bindings for interaction
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<MouseWheel>", self._on_wheel)   # Win/mac
        self.bind("<Button-4>", self._on_wheel)     # Linux up
        self.bind("<Button-5>", self._on_wheel)     # Linux down
        self.objp = None
        self.T_w_cam0 = None
        self.T_w_cam1 = None
        self._custom_view = None  # (eye, center, up) for camera view
        
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw   = 0.0     # degrees, around +Y
        self.pitch = 20.0    # degrees, up/down
        self.radius = 4.0    # distance from center
        self._orbit_speed = 0.3   # deg per pixel
        self._pan_speed   = 0.001 # world-units per pixel (scaled by radius)
        self._zoom_speed  = 0.0015

    def set_view_to_camera(self, T_w_cam):
        if T_w_cam is None:
            return
        R_wc = T_w_cam[:3, :3]
        t_wc = T_w_cam[:3, 3]
        # Camera forward dir in world (OpenCV camera looks +Z)

        forward_world = R_wc @ np.array([0, 0, 1], dtype=np.float32)   # +Z_c
        up_world = R_wc @ np.array([0,-1, 0], dtype=np.float32)   # -Y_c (since +Y_c is down)

        # set orbit params from pose
        self.radius = getattr(self, "radius", 1.0) or 1.0
        self.center = t_wc + forward_world * self.radius
        
        f = forward_world / (np.linalg.norm(forward_world)+1e-9)
        self.pitch = float(np.rad2deg(np.arcsin(f[1])))
        self.yaw   = float(np.rad2deg(np.arctan2(f[0], f[2])))

        # store a custom up if you want to keep camera roll:
        self._up_override = up_world  # optional; else use (0,1,0)

        self.redraw()
        
    def reset_view(self):
        self._custom_view = None
        self.after_idle(self.redraw)
    
    def _eye_center_up(self):
        # yaw, pitch in radians
        ya = np.deg2rad(self.yaw)
        pa = np.deg2rad(self.pitch)
        # forward in world coords (OpenGL-style Y up)
        fx =  np.sin(ya) * np.cos(pa)
        fy =  np.sin(pa)
        fz =  np.cos(ya) * np.cos(pa)
        forward = np.array([fx, fy, fz], dtype=np.float32)
        eye = self.center + (-forward) * self.radius  # look from opposite of forward
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up = getattr(self, "_up_override", None)
        if up is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # world Y-up default

        return eye, self.center, up

    def _on_press(self, e):
        self._last = (e.x, e.y)
        self._drag_btn = e.num if hasattr(e, "num") else 1  # 1=LMB, 2=MMB, 3=RMB

    def _on_drag(self, e):
        if not self._last: return
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        self._last = (e.x, e.y)

        # Shift = pan, otherwise orbit with LMB; RMB/MMB can be pan if you prefer
        is_pan = (e.state & 0x0001) != 0 or self._drag_btn in (2,3)  # Shift or MMB/RMB

        if is_pan:
            # Pan in camera plane
            eye, center, up = self._eye_center_up()
            f = (center - eye); f = f / (np.linalg.norm(f) + 1e-9)
            right = np.cross(f, up); right /= (np.linalg.norm(right)+1e-9)
            upv = np.cross(right, f)
            s = self._pan_speed * self.radius
            self.center += (-dx * s) * right + (dy * s) * upv
        else:
            # Orbit around center
            self.yaw   += dx * self._orbit_speed
            self.pitch += dy * self._orbit_speed
            self.pitch = float(np.clip(self.pitch, -89.0, 89.0))

        self.after_idle(self.redraw)

    def _on_wheel(self, e):
        delta = getattr(e, "delta", 0)
        if delta == 0:
            delta = 120 if getattr(e, "num", 0) == 4 else -120  # X11 buttons 4/5
        # exponential zoom feels nicer; clamp radius
        self.radius *= float(np.exp(-delta * self._zoom_speed))
        self.radius = float(np.clip(self.radius, 0.2, 100.0))
        self.after_idle(self.redraw)
        
    def redraw(self):
        self.tkMakeCurrent()
        w, h = max(self.width, 1), max(self.height, 1)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        eye, center, up = self._eye_center_up()
        gluLookAt(*eye, *center, *up)
        
        if self.objp is not None:
            # Draw object points as blue circles (disks)
            glColor3f(0.2, 0.4, 1.0)
            for p in self.objp:
                glPushMatrix()
                glTranslatef(p[0], p[1], p[2])
                quad = gluNewQuadric()
                gluDisk(quad, 0, 0.01, 16, 1)  # radius=0.04, blue disk
                gluDeleteQuadric(quad)
                glPopMatrix()
        # Axes
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)  # X
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)  # Y
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)  # Z
        glEnd()

        # Draw camera coordinate frames if available
        def draw_axes(T, name="", length=0.2):
            if T is None:
                return
            origin = T[:3, 3]
            R = T[:3, :3]
            axes = np.eye(3) * length
            glLineWidth(3.0)
            glBegin(GL_LINES)
            # X axis (red)
            glColor3f(1, 0, 0)
            glVertex3fv(origin)
            glVertex3fv(origin + R @ axes[:, 0])
            # Y axis (green)
            glColor3f(0, 1, 0)
            glVertex3fv(origin)
            glVertex3fv(origin + R @ axes[:, 1])
            # Z axis (blue)
            glColor3f(0, 0, 1)
            glVertex3fv(origin)
            glVertex3fv(origin + R @ axes[:, 2])
            glEnd()
            glLineWidth(1.0)
            # Draw the name of the coordinate frame
            if name:
                glColor3f(1, 1, 1)
                glRasterPos3fv(origin + np.array([0, 0, length * 1.2]))
                for ch in name:
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

        draw_axes(self.T_w_cam0, name="C0")
        draw_axes(self.T_w_cam1, name="C1")

        # Swap buffers (version-safe)
        if hasattr(self, "swap_buffers"):
            self.swap_buffers()
        elif hasattr(self, "tkSwapBuffers"):
            self.tkSwapBuffers()
        else:
            glFlush()
            
class CalibrationUI(tk.Tk):
    def __init__(self, img_path, K0, D0, h0, w0, K1, D1, h1, w1, GUI_H=1200, GUI_W=1800, chessboard_size=(8, 6), square_size=0.06, cam_dict={0: 'realsense', 1: 'tof'}, undistorted=False):
        super().__init__()
        self.title("StereoXtrinsicsUI")
        self.geometry(f"{GUI_W}x{GUI_H}")
        self.GUI_W = GUI_W
        self.GUI_H = GUI_H
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=2)

        image_path_0 = glob.glob(os.path.join(img_path, 'cam0/*.png'))
        image_path_1 = glob.glob(os.path.join(img_path, 'cam1/*.png'))
        self.image_pairs = list(zip(sorted(image_path_0), sorted(image_path_1)))
        self.index = 0
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        self.objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
        self.objp = self.objp * square_size

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        self.K0 = np.array(K0)
        self.D0 = np.array(D0)
        self.K1 = np.array(K1)
        self.D1 = np.array(D1)
        
        # self.K0 = np.array([[917.626363, 0.000000, 631.016065],
        #                     [0.000000, 920.528195, 345.491278],
        #                     [0.000000, 0.000000, 1.000000]]
        #                     )
        
        # self.D0 = np.array([0.082749, -0.124803, -0.001593, -0.000622, 0.000000])
        
        
        # self.K1 = np.array([[205.731476, 0.000000, 112.496540],
        #                 [0.000000, 206.057320, 85.766603],
        #                 [0.000000, 0.000000, 1.000000]]
        #                )
        # self.D1 = np.array([0.287139, -0.740035, 0.000383, -0.001975, 0.000000])
        
        # Original Flexx2 ToF
        # self.K1 = np.array([[205.2405242919922, 0.000000, 112.6792221069336],
        #                 [0.000000, 205.2405242919922, 86.31165313720703],
        #                 [0.000000, 0.000000, 1.000000]]
        #                )
        
        # self.D1 = np.array([0.33247798681259155, -1.115235447883606, -0.0020122069399803877, 0.0008649492519907653, 0.6790379881858826])
        
        self.h0, self.w0 = h0, w0
        self.h1, self.w1 = h1, w1
        
        self.undistorted = undistorted
        if self.undistorted:
            self.mapx0, self.mapy0 = cv2.initUndistortRectifyMap(self.K0, self.D0, None, self.K0, (self.w0,self.h0), m1type=cv2.CV_16SC2)
            self.mapx1, self.mapy1 = cv2.initUndistortRectifyMap(self.K1, self.D1, None, self.K1, (self.w1,self.h1), m1type=cv2.CV_16SC2)
            self.D0 = np.zeros_like(self.D0)
            self.D1 = np.zeros_like(self.D1)

        top = tk.Frame(self)
        left = tk.Frame(self)
        right = tk.Frame(self)
        
        top.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        left.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        right.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

        # Top controls
        # top = tk.Frame(left, pady=6)
        # top.pack(side=tk.TOP, fill=tk.X)
        self.next_frame_btn = tk.Button(top, text="Next Frame ▶", command=self.next_frame)
        self.next_frame_btn.pack(side=tk.LEFT, padx=0)

        # self.detect_corners_btn = tk.Button(top, text="Detect Corners", command=self.detect_corners)
        # self.detect_corners_btn.pack(side=tk.LEFT, padx=12)

        self.toggle_corners_btn = tk.Button(top, text="Toggle Corners", command=self.toggle_corners)
        self.toggle_corners_btn.pack(side=tk.LEFT, padx=0)

        self.add_btn = tk.Button(top, text="Add", command=self.add)
        self.add_btn.pack(side=tk.LEFT, padx=0)

        self.calibrate_btn = tk.Button(top, text="Calibrate", command=self.calibrate)
        self.calibrate_btn.pack(side=tk.LEFT, padx=0)
        
        self.save_btn = tk.Button(top, text="Save", command=self.save_calib)
        self.save_btn.pack(side=tk.LEFT, padx=0)

        
        
        # Add OpenGL panel on the right side
        # right_frame = tk.Frame(right)
        # right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(
            # right_frame,
            right,
            text=f"Coordinate Frame View",
        ).pack(anchor="center", pady=(0,8))
        
        self.max_height = self.GUI_H // 4
        scale0 = self.max_height / self.h0
        scale1 = self.max_height / self.h1
        
        resized_w0 = self.w0 * scale0
        resized_w1 = self.w1 * scale1
        
        gl_viewer_w = self.GUI_W - resized_w0 - resized_w1
        # gl_viewer_w = resized_w0 + resized_w1
        # self.gl = GLPanel(right_frame, width=gl_viewer_w - 100, height=self.max_height)
        self.gl = GLPanel(right, width=gl_viewer_w, height=int(self.GUI_H * 0.8))
        self.gl.pack(fill=tk.BOTH, expand=True)
        # Add buttons to switch view to C0 or C1
        # self.view_c0_btn = tk.Button(top, text="View from C0", command=lambda: self.gl.set_view_to_camera(self.gl.T_w_cam0))
        # self.view_c0_btn.pack(side=tk.LEFT, padx=12)
        # self.view_c1_btn = tk.Button(top, text="View from C1", command=lambda: self.gl.set_view_to_camera(self.gl.T_w_cam1))
        # self.view_c1_btn.pack(side=tk.LEFT, padx=12)
        # self.reset_view_btn = tk.Button(top, text="Reset View", command=self.gl.reset_view)
        # self.reset_view_btn.pack(side=tk.LEFT, padx=12)
        
        self.fix_view_on_c0 = tk.BooleanVar(value=False)
        self.view_c0_toggle_checkbtn = tk.Checkbutton(top, text="View from C0", variable=self.fix_view_on_c0, command=self.on_toggle_view_c0)
        self.view_c0_toggle_checkbtn.pack(side=tk.LEFT, padx=0)
        
        self.fix_view_on_c1 = tk.BooleanVar(value=False)
        self.view_c1_toggle_checkbtn = tk.Checkbutton(top, text="View from C1", variable=self.fix_view_on_c1, command=self.on_toggle_view_c1)
        self.view_c1_toggle_checkbtn.pack(side=tk.LEFT, padx=0)
        
        
        # Place label below the buttons, centered
        # label_frame = tk.Frame(top)
        label_frame = tk.Frame(left)
        label_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.header_raw_images_panel = tk.Label(
            label_frame,
            text=f"Raw Image Pairs {cam_dict[0]} | {cam_dict[1]}",
            # font=("TkDefaultFont", 12, "bold")
        )
        self.header_raw_images_panel.pack(anchor="center", pady=(0,8))
        
        
        
        
        # self.image_label = tk.Label(top, image=None)
        self.image_label = tk.Label(left, image=None)
        self.image_label.image = None  # Keep a reference!
        self.image_label.pack(anchor='nw', expand=True)
        # ttk.Separator(top, orient="horizontal").pack(fill="x", pady=8)
        # mid = tk.Frame(top, pady=6)
        mid = tk.Frame(left, pady=6)
        mid.pack(side=tk.TOP, fill=tk.X)
        
        # label_frame_added_images = tk.Frame(top)
        label_frame_added_images = tk.Frame(left)
        label_frame_added_images.pack(side=tk.TOP, fill=tk.X)
        self.header_added_images_panel = tk.Label(
            label_frame_added_images,
            text=f"Added Image Pairs {cam_dict[0]} | {cam_dict[1]}",
            # font=("TkDefaultFont", 12, "bold")
        )
        self.header_added_images_panel.pack(anchor="center", pady=(0,8))
        
        # self.image_added_label = tk.Label(top, image=None)
        self.image_added_label = tk.Label(left, image=None)
        self.image_added_label.image = None  # Keep a reference!
        self.image_added_label.pack(anchor="nw", expand=True)
            
        self.prev_added_frame_btn = tk.Button(mid, text="Left", command=self.prev_added_frame)
        self.prev_added_frame_btn.pack(side=tk.LEFT, padx=0)
        
        self.next_added_frame_btn = tk.Button(mid, text="Right", command=self.next_added_frame)
        self.next_added_frame_btn.pack(side=tk.LEFT, padx=0)
        
        self.delete_added_frame_btn = tk.Button(mid, text="Delete", command=self.delete_added_frame)
        self.delete_added_frame_btn.pack(side=tk.LEFT, padx=0)
        
        self.disply_added_frame_id = 0
        
        self.img0 = None
        self.img1 = None

        self.objpoints3d = [] # 3d point in real world space
        self.imgpoints0 = [] # 2d points in image plane.
        self.imgpoints1 = [] # 2d points in image plane.
        self.image_added = []
        
        self.is_calibrated = False

        self.rvecs = []
        self.tvecs = []
        
        self.objp3d_vis = None
    
    def on_toggle_view_c0(self):
        if self.fix_view_on_c0.get():              # toggled ON
            self.gl.set_view_to_camera(self.gl.T_w_cam0)
        self.fix_view_on_c1.set(False)
        
    def on_toggle_view_c1(self):
        if self.fix_view_on_c1.get():              # toggled ON
            self.gl.set_view_to_camera(self.gl.T_w_cam1)
        self.fix_view_on_c0.set(False)
        
    def save_calib(self):
        if not self.is_calibrated:
            messagebox.showerror("Error", "Please calibrate before saving.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if not save_path:
            return
        calib_data = {
            'K0': self.K0.tolist(),
            'D0': self.D0.tolist(),
            'K1': self.K1.tolist(),
            'D1': self.D1.tolist(),
            'R': self.rvec_2_1.tolist(),
            'T': self.tvec_2_1.reshape(3).tolist(),
            'chessboard_size': [self.chessboard_size[0], self.chessboard_size[1]],
            'square_size': self.square_size
        }
        with open(save_path, 'w') as f:
            yaml.dump(calib_data, f)
        messagebox.showinfo("Saved", f"Calibration data saved to {save_path}")
    
    def delete_added_frame(self):
        if not self.image_added:
            print("No image pairs added yet.")
            return
        self.image_added.pop(self.disply_added_frame_id)
        self.objpoints3d.pop(self.disply_added_frame_id)
        self.imgpoints0.pop(self.disply_added_frame_id)
        self.imgpoints1.pop(self.disply_added_frame_id)
        self.rvecs.pop(self.disply_added_frame_id)
        self.tvecs.pop(self.disply_added_frame_id)
        
        if len(self.image_added) == 0:
            self.disply_added_frame_id = 0
            self.image_added_label.configure(image=None)
            self.image_added_label.image = None
            self.header_added_images_panel.configure(text=f"Added Image Pairs 0 / 0")
            self.gl.objp = None
            self.gl.T_w_cam0 = None
            self.gl.T_w_cam1 = None
            
            self.gl.redraw()
            return
        
        self.disply_added_frame_id = self.disply_added_frame_id % len(self.image_added)
        self.render_added_frame()
         
    def prev_added_frame(self):
        if not self.image_added:
            print("No image pairs added yet.")
            return
        self.disply_added_frame_id -= 1
        self.disply_added_frame_id = self.disply_added_frame_id % len(self.image_added)
        if self.disply_added_frame_id < 0:
            self.disply_added_frame_id = len(self.image_added) - 1

        self.render_added_frame()
    
    def next_added_frame(self):
        if not self.image_added:
            print("No image pairs added yet.")
            return
        self.disply_added_frame_id += 1
        self.disply_added_frame_id = self.disply_added_frame_id % len(self.image_added)
        
        self.render_added_frame()

    def render_added_frame(self):
        self.header_added_images_panel.configure(text=f"Added Image Pairs {self.disply_added_frame_id + 1} / {len(self.image_added)}")
        # Resize images to fit in the window if necessary
        img0, img1 = self.image_added[self.disply_added_frame_id][0].copy(), self.image_added[self.disply_added_frame_id][1].copy()
        
        cv2.drawChessboardCorners(img0, self.chessboard_size, self.imgpoints0[self.disply_added_frame_id], True)
        cv2.drawChessboardCorners(img1, self.chessboard_size, self.imgpoints1[self.disply_added_frame_id], True)
        
        self.gl.objp = self.objpoints3d[self.disply_added_frame_id]
        R, _ = cv2.Rodrigues(self.rvecs[self.disply_added_frame_id])
        t = self.tvecs[self.disply_added_frame_id].reshape(3)
        T_c0_w = self.make_transform(R, t)
        T_w_c0 = np.linalg.inv(T_c0_w)
        
        self.gl.T_w_cam0 = T_w_c0
        
        if self.is_calibrated:
            preproj0, _ = cv2.projectPoints(self.objpoints3d[self.disply_added_frame_id], self.rvecs[self.disply_added_frame_id], self.tvecs[self.disply_added_frame_id], self.K0, self.D0)
            self.objpoints3d[self.disply_added_frame_id]
            # R, _ = cv2.Rodrigues(self.rvecs[self.disply_added_frame_id])
            # t = self.tvecs[self.disply_added_frame_id].reshape(3)
            # T_c0_w = self.make_transform(R, t)
            # Transform object points from world to camera 1 coordinates
            p3d_w = self.objpoints3d[self.disply_added_frame_id]
            # Convert to homogeneous coordinates
            p3d_w_h = np.hstack([p3d_w, np.ones((p3d_w.shape[0], 1))])
            # Transform to camera 1 frame
            p3d_c1_h = (T_c0_w @ p3d_w_h.T).T
            p3d_c1 = p3d_c1_h[:, :3]
            preproj1, _ = cv2.projectPoints(p3d_c1, self.rvec_2_1, self.tvec_2_1, self.K1, self.D1)

            # Compute T_w_c1 using self.rvec_2_1 (3x3 rotation) and self.tvec_2_1 (3x1 translation)
            T_c1_c0 = self.make_transform(self.rvec_2_1, self.tvec_2_1.reshape(3))
            T_c0_c1 = np.linalg.inv(T_c1_c0)
            T_w_c1 = T_w_c0 @ T_c0_c1
            self.gl.T_w_cam1 = T_w_c1
            # Draw reprojected points as red circles on img1
            for pt in preproj0.squeeze().astype(int):
                cv2.circle(img0, tuple(pt), 3, (0, 0, 255), -1)
                
            # Draw reprojected points as red circles on img1
            for pt in preproj1.squeeze().astype(int):
                cv2.circle(img1, tuple(pt), 3, (0, 0, 255), -1)
            
            # Compute RMS reprojection error for both cameras
            reproj0 = preproj0.squeeze()
            reproj1 = preproj1.squeeze()
            gt0 = self.imgpoints0[self.disply_added_frame_id].squeeze()
            gt1 = self.imgpoints1[self.disply_added_frame_id].squeeze()
            err0 = np.linalg.norm(reproj0 - gt0, axis=1)
            err1 = np.linalg.norm(reproj1 - gt1, axis=1)
            rms0 = np.sqrt(np.mean(err0 ** 2))
            rms1 = np.sqrt(np.mean(err1 ** 2))
            print(f"RMS reprojection error (cam0): {rms0:.3f} px, (cam1): {rms1:.3f} px")
            
        max_height = self.max_height
        scale0 = max_height / img0.shape[0]
        scale1 = max_height / img1.shape[0]
        img0 = cv2.resize(img0, (int(img0.shape[1] * scale0), max_height))
        img1 = cv2.resize(img1, (int(img1.shape[1] * scale1), max_height))
        
        # max_width = self.GUI_W // 3
        # scale0 = max_width / img0.shape[1]
        # scale1 = max_width / img1.shape[1]
        # img0 = cv2.resize(img0, (max_width, int(img0.shape[0] * scale0)))
        # img1 = cv2.resize(img1, (max_width, int(img1.shape[0] * scale1)))
        
        # img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        # Combine images side by side
        combined_img = np.hstack((img0, img1))
        
        # Convert to PhotoImage
        # combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", combined_img)
        io_buf = io.BytesIO(buffer)
        photo = tk.PhotoImage(data=io_buf.getvalue())
            
        if hasattr(self, 'image_added_label'):
            self.image_added_label.config(image=photo)
            self.image_added_label.image = photo  # Keep a reference!
        else:
            self.image_added_label = tk.Label(self, image=photo)
            self.image_added_label.image = photo  # Keep a reference!
            self.image_added_label.pack(side=tk.TOP, expand=True)
        
        if self.fix_view_on_c0.get():             
            self.gl.set_view_to_camera(self.gl.T_w_cam0)
        if self.fix_view_on_c1.get():             
            self.gl.set_view_to_camera(self.gl.T_w_cam1)
        self.gl.redraw()
    def calibrate(self):
        print("Calibrating...")
        print("Number of image pairs:", len(self.objpoints3d))
        # if self.undistorted:
        retval, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints3d, self.imgpoints0, self.imgpoints1,
            self.K0, self.D0, self.K1, self.D1, (self.w0, self.h0),
            flags=(cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
        )
        # else:
        #     retval, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        #         self.objpoints3d, self.imgpoints0, self.imgpoints1,
        #         self.K0, self.D0, self.K1, self.D1, (640, 480),
        #         flags=(cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS),
        #         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
        #     )
        
        print("Stereo Calibration RMS error:", retval)
        print("Translation:\n", T)
        print("Rotation:\n", R)
        print("K0:\n", K1)
        print("D0:\n", d1)
        print("K1:\n", K2)
        print("D1:\n", d2)
        
        self.tvec_2_1 = T
        self.rvec_2_1 = R
        self.is_calibrated = True
        self.render_added_frame()
    def add(self):
        self.objpoints3d.append(self.objp)
        self.imgpoints0.append(self.corners0)
        self.imgpoints1.append(self.corners1)
        self.image_added.append((self.img0_raw.copy(), self.img1_raw.copy()))
        success, rvec, tvec = cv2.solvePnP(
            self.objp, self.corners0, self.K0, self.D0, flags=cv2.SOLVEPNP_ITERATIVE
        )
        self.rvecs.append(rvec)
        self.tvecs.append(tvec)
        self.next_frame()
        self.render_added_frame()
        
    def toggle_corners(self):
        self.corners1 = self.corners1[::-1]
        self.img1 = self.img1_raw.copy()
        cv2.drawChessboardCorners(self.img1, self.chessboard_size, self.corners1, True)
        self.render()
        
    def detect_corners(self):
        self.img0_raw = self.img0.copy()
        self.img1_raw = self.img1.copy()   
        
        gray0 = cv2.cvtColor(self.img0_raw, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(self.img1_raw, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret0, corners0 = cv2.findChessboardCorners(gray0, self.chessboard_size, None)
        ret1, corners1 = cv2.findChessboardCorners(gray1, self.chessboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret1 == True and ret0 == True:
            self.corners0 = cv2.cornerSubPix(gray0, corners0, (11,11), (-1,-1), self.criteria)
            self.corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), self.criteria)
            # corners2 = corners2[::-1]
            cv2.drawChessboardCorners(self.img0, self.chessboard_size, self.corners0, ret0)
            cv2.drawChessboardCorners(self.img1, self.chessboard_size, self.corners1, ret1)
            # cv2.imshow('img', img0)
            # cv2.waitKey(0)
        self.render()
        
    def next_frame(self):
        if not self.image_pairs:
            print("No image pairs found.")
            return
        
        self.index = self.index % len(self.image_pairs)
        img_path_0, img_path_1 = self.image_pairs[self.index]
        self.index += 1
        
        self.img0 = cv2.imread(img_path_0)
        self.img1 = cv2.imread(img_path_1)
        
        if self.undistorted:
            self.img0 = cv2.remap(self.img0, self.mapx0, self.mapy0, interpolation=cv2.INTER_LINEAR)
            self.img1 = cv2.remap(self.img1, self.mapx1, self.mapy1, interpolation=cv2.INTER_LINEAR)
        # print(self.img0.shape)
        # print(self.img1.shape)
        
        self.header_raw_images_panel.configure(text=f"Raw Image Pairs {img_path_0.split('/')[-2:][0]}/{img_path_0.split('/')[-2:][1]} | {img_path_1.split('/')[-2:][0]}/{img_path_1.split('/')[-2:][1]}")
        self.detect_corners()
        self.render()
        
    def render(self):
        # Resize images to fit in the window if necessary
        # max_height = 400
        max_height = self.max_height
        img0 = self.img0
        img1 = self.img1
        scale0 = max_height / img0.shape[0]
        scale1 = max_height / img1.shape[0]
        img0 = cv2.resize(img0, (int(img0.shape[1] * scale0), max_height))
        img1 = cv2.resize(img1, (int(img1.shape[1] * scale1), max_height))
        
        # max_width = self.GUI_W // 3
        # scale0 = max_width / img0.shape[1]
        # scale1 = max_width / img1.shape[1]
        # img0 = cv2.resize(img0, (max_width, int(img0.shape[0] * scale0)))
        # img1 = cv2.resize(img1, (max_width, int(img1.shape[0] * scale1)))
        
        # img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        # Combine images side by side
        combined_img = np.hstack((img0, img1))
        
        # Convert to PhotoImage
        # combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", combined_img)
        io_buf = io.BytesIO(buffer)
        photo = tk.PhotoImage(data=io_buf.getvalue())
        
        # Display image
        if hasattr(self, 'image_label'):
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference!
        else:
            self.image_label = tk.Label(self, image=photo)
            self.image_label.image = photo  # Keep a reference!
            self.image_label.pack(side=tk.TOP, expand=True)
            
    def make_transform(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StereoXtrinsics UI launcher"
    )
    parser.add_argument(
        "--config", "-c",
        default='./config/default.yaml',
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--width", "-W",
        default=1920,
        type=int,
        help="GUI window width (overrides YAML)"
    )
    parser.add_argument(
        "--height", "-H",
        default=1080,
        type=int,
        help="GUI window height (overrides YAML)"
    )

    args = parser.parse_args()

    # Resolve config path and read YAML
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with cfg_path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Failed to parse YAML: {e}", file=sys.stderr)
            sys.exit(1)
    
    config['GUI_W'] = args.width
    config['GUI_H'] = args.height
    
    CalibrationUI(**config).mainloop()