// Three.js scene setup and initialization

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { threeState } from '../state';

// Create 3D grid and axes
export function createGridAndAxes(): void {
  if (!threeState.scene) return;

  // Remove existing
  if (threeState.gridHelper) {
    threeState.scene.remove(threeState.gridHelper);
    threeState.gridHelper.dispose();
  }
  if (threeState.axesGroup) {
    threeState.scene.remove(threeState.axesGroup);
  }

  // Create subtle grid on the "floor" (XZ plane)
  threeState.gridHelper = new THREE.GridHelper(50, 20, 0x333333, 0x222222);
  threeState.gridHelper.position.y = -25;
  threeState.scene.add(threeState.gridHelper);

  // Create axes with labels
  threeState.axesGroup = new THREE.Group();

  const axisLength = 25;
  const axisColors = {
    x: 0xff4444, // red
    y: 0x44ff44, // green
    z: 0x4444ff, // blue
  };

  // Create axis lines
  const createAxis = (
    start: THREE.Vector3,
    end: THREE.Vector3,
    color: number
  ) => {
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    const material = new THREE.LineBasicMaterial({
      color,
      opacity: 0.5,
      transparent: true,
    });
    return new THREE.Line(geometry, material);
  };

  // X axis (red)
  threeState.axesGroup.add(
    createAxis(
      new THREE.Vector3(-axisLength, -25, 0),
      new THREE.Vector3(axisLength, -25, 0),
      axisColors.x
    )
  );

  // Y axis (green)
  threeState.axesGroup.add(
    createAxis(
      new THREE.Vector3(0, -25, 0),
      new THREE.Vector3(0, axisLength, 0),
      axisColors.y
    )
  );

  // Z axis (blue)
  threeState.axesGroup.add(
    createAxis(
      new THREE.Vector3(0, -25, -axisLength),
      new THREE.Vector3(0, -25, axisLength),
      axisColors.z
    )
  );

  threeState.scene.add(threeState.axesGroup);
}

// Create selection box overlay
export function createSelectionBox(container: HTMLElement): void {
  threeState.selectionBox = document.createElement('div');
  threeState.selectionBox.style.cssText = `
    position: absolute;
    border: 2px solid #3b82f6;
    background: rgba(59, 130, 246, 0.1);
    pointer-events: none;
    display: none;
    z-index: 100;
  `;
  container.appendChild(threeState.selectionBox);
}

// Initialize Three.js scene
export function initThree(container: HTMLElement): void {
  threeState.scene = new THREE.Scene();
  threeState.scene.background = new THREE.Color(0x0a0a0a);

  threeState.camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  threeState.camera.position.z = 50;

  threeState.renderer = new THREE.WebGLRenderer({ antialias: true });
  threeState.renderer.setSize(container.clientWidth, container.clientHeight);
  threeState.renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(threeState.renderer.domElement);

  threeState.controls = new OrbitControls(
    threeState.camera,
    threeState.renderer.domElement
  );
  threeState.controls.enableDamping = true;
  threeState.controls.dampingFactor = 0.05;

  threeState.raycaster = new THREE.Raycaster();
  threeState.raycaster.params.Points!.threshold = 0.5;

  // Add grid and axes
  createGridAndAxes();

  // Create selection box overlay
  createSelectionBox(container);

  // Handle window resize
  window.addEventListener('resize', () => {
    if (!threeState.camera || !threeState.renderer) return;
    threeState.camera.aspect = container.clientWidth / container.clientHeight;
    threeState.camera.updateProjectionMatrix();
    threeState.renderer.setSize(container.clientWidth, container.clientHeight);
  });
}

// Render loop
export function render(): void {
  if (!threeState.renderer || !threeState.scene || !threeState.camera) return;
  threeState.renderer.render(threeState.scene, threeState.camera);
}

// Update controls
export function updateControls(): void {
  threeState.controls?.update();
}

// Reset camera position
export function resetCamera(): void {
  if (!threeState.camera || !threeState.controls) return;
  threeState.camera.position.set(0, 0, 50);
  threeState.controls.reset();
}

// Get renderer DOM element
export function getRendererElement(): HTMLCanvasElement | null {
  return threeState.renderer?.domElement ?? null;
}
