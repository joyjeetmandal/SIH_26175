import React, { useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { uploadImage } from "./api";
import "./style.css";


function TerrainViewer({ 
  heightmapUrl,
  metadataUrl,
  heightmapData,
 }) {
  const mountRef = React.useRef(null);

  const [hoverInfo, setHoverInfo] = React.useState(null);

  React.useEffect(() => {
    if (!heightmapUrl) return;

    const mount = mountRef.current;

    // =====================================================
    // SCENE
    // =====================================================

    const scene = new THREE.Scene();

    scene.background = new THREE.Color(0x111111);

    // =====================================================
    // CAMERA
    // =====================================================

    const camera = new THREE.PerspectiveCamera(
      45,
      mount.clientWidth / mount.clientHeight,
      0.1,
      5000
    );

    camera.position.set(
      0,
      180,
      300
    );

    // =====================================================
    // RENDERER
    // =====================================================

    const renderer =
      new THREE.WebGLRenderer({
        antialias: true,
      });

    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, 2)
    );

    renderer.setSize(
      mount.clientWidth,
      mount.clientHeight
    );

    renderer.outputColorSpace =
      THREE.SRGBColorSpace;

    mount.appendChild(
      renderer.domElement
    );

    // =====================================================
    // LIGHTING
    // =====================================================

    const ambientLight =
      new THREE.AmbientLight(
        0xffffff,
        1.8
      );

    scene.add(ambientLight);

    const directionalLight =
      new THREE.DirectionalLight(
        0xffffff,
        1.5
      );

    directionalLight.position.set(
      200,
      400,
      200
    );

    scene.add(
      directionalLight
    );

    // =====================================================
    // LOAD HEIGHTMAP
    // =====================================================

    const imageLoader =
      new THREE.ImageLoader();

    imageLoader.load(
      heightmapUrl,
      (image) => {

        // =================================================
        // DOWNSAMPLE
        // Similar to Python max_dimension = 350
        // =================================================

        if (!heightmapData || !heightmapData.length) {
  return;
}

const maxDimension = 250;

// Original heightmap dimensions
const height = heightmapData.length;
const width = heightmapData[0].length;

// Downsample for Three.js performance
const step = Math.max(
  1,
  Math.ceil(Math.max(height, width) / maxDimension)
);

// Create visual heightmap from REAL elevation data
const visualHeightmap = [];

for (let y = 0; y < height; y += step) {
  const row = [];

  for (let x = 0; x < width; x += step) {
    row.push(heightmapData[y][x]);
  }

  visualHeightmap.push(row);
}

const visualHeight = visualHeightmap.length;
const visualWidth = visualHeightmap[0].length;

console.log("Original heightmap:", width, "x", height);
console.log(
  "Visual heightmap:",
  visualWidth,
  "x",
  visualHeight
);


        // =================================================
        // TERRAIN GEOMETRY
        // =================================================

        const terrainWidth = 400;
const terrainDepth = 400;

// Use REAL elevation values directly.
const heightScale = 1;

const geometry = new THREE.PlaneGeometry(
  terrainWidth,
  terrainDepth,
  visualWidth - 1,
  visualHeight - 1
);

const positions = geometry.attributes.position.array;

for (let y = 0; y < visualHeight; y++) {
  for (let x = 0; x < visualWidth; x++) {
    const index = y * visualWidth + x;

    const elevation =
      Number(visualHeightmap[y][x]) * heightScale;

    positions[index * 3 + 2] = elevation;
  }
}

geometry.attributes.position.needsUpdate = true;
geometry.computeVertexNormals();

        // =================================================
        // JET COLOR FUNCTION
        // =================================================

        function jetColor(value) {

          value =
            THREE.MathUtils.clamp(
              value,
              0,
              1
            );

          let r = 0;
          let g = 0;
          let b = 0;

          if (value < 0.25) {

            r = 0;
            g =
              value / 0.25;
            b = 1;

          } else if (
            value < 0.5
          ) {

            r = 0;
            g = 1;
            b =
              1 -
              (value - 0.25) /
                0.25;

          } else if (
            value < 0.75
          ) {

            r =
              (value - 0.5) /
              0.25;

            g = 1;
            b = 0;

          } else {

            r = 1;

            g =
              1 -
              (value - 0.75) /
                0.25;

            b = 0;
          }

          return new THREE.Color(
            r,
            g,
            b
          );
        }

        // =================================================
// VERTEX COLORS
// =================================================

const colors = new Float32Array(
  visualWidth * visualHeight * 3
);

// Find actual elevation range
let minElevation = Infinity;
let maxElevation = -Infinity;

for (let y = 0; y < visualHeight; y++) {
  for (let x = 0; x < visualWidth; x++) {
    const elevation = Number(
      visualHeightmap[y][x]
    );

    minElevation = Math.min(
      minElevation,
      elevation
    );

    maxElevation = Math.max(
      maxElevation,
      elevation
    );
  }
}

const elevationRange =
  maxElevation - minElevation;

// Assign colors to every vertex
for (let y = 0; y < visualHeight; y++) {
  for (let x = 0; x < visualWidth; x++) {
    const index =
      y * visualWidth + x;

    const elevation = Number(
      visualHeightmap[y][x]
    );

    let normalized = 0;

    if (elevationRange > 0) {
      normalized =
        (elevation - minElevation) /
        elevationRange;
    }

    const color =
      jetColor(normalized);

    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  }
}

geometry.setAttribute(
  "color",
  new THREE.BufferAttribute(
    colors,
    3
  )
);

        // =================================================
        // MATERIAL
        // =================================================

        const material =
          new THREE.MeshStandardMaterial({
            vertexColors: true,

            side:
              THREE.DoubleSide,

            roughness: 0.9,

            metalness: 0.0,

            wireframe: false,
          });

        // =================================================
        // TERRAIN
        // =================================================

        const terrain =
          new THREE.Mesh(
            geometry,
            material
          );

        terrain.name = "terrain";

        /*
         * Make:
         *
         * X = X
         * Y = elevation
         * Z = Y
         */

        terrain.rotation.x =
          -Math.PI / 2;

        scene.add(
          terrain
        );

        // =================================================
        // GRID
        // =================================================

        const grid =
          new THREE.GridHelper(
            terrainWidth,
            20,
            0x777777,
            0x333333
          );

        grid.position.y = 0;

        scene.add(
          grid
        );

        // =================================================
        // AXES
        // =================================================

        const axes =
          new THREE.AxesHelper(
            220
          );

        axes.position.y = 1;

        scene.add(
          axes
        );

        // =================================================
        // ORBIT CONTROLS
        // =================================================

        const controls =
          new OrbitControls(
            camera,
            renderer.domElement
          );

        // =====================================================
// MOUSE HOVER / RAYCASTING
// =====================================================

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

const handleMouseMove = (event) => {
  if (!heightmapData || !terrain) {
    return;
  }

  const rect = renderer.domElement.getBoundingClientRect();

  mouse.x =
    ((event.clientX - rect.left) / rect.width) * 2 - 1;

  mouse.y =
    -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  const intersects = raycaster.intersectObject(
    terrain,
    false
  );

  if (intersects.length === 0) {
    setHoverInfo(null);
    return;
  }

  const intersection = intersects[0];

  // Convert intersection point into terrain's local coordinates
  const localPoint = terrain.worldToLocal(
    intersection.point.clone()
  );

  // Original heightmap dimensions
  const height = heightmapData.length;
  const width = heightmapData[0].length;

  // Convert local X position to matrix X
  const matrixX = THREE.MathUtils.clamp(
    Math.round(
      ((localPoint.x + terrainWidth / 2) /
        terrainWidth) *
        (width - 1)
    ),
    0,
    width - 1
  );

  // PlaneGeometry rows run from +Y to -Y.
  // This keeps matrix Y=0 at the top of the heightmap.
  const matrixY = THREE.MathUtils.clamp(
    Math.round(
      ((terrainDepth / 2 - localPoint.y) /
        terrainDepth) *
        (height - 1)
    ),
    0,
    height - 1
  );

  // GET THE ACTUAL ELEVATION FROM heightmap.npy
  const elevation = Number(
    heightmapData[matrixY][matrixX]
  );

  setHoverInfo({
    x: matrixX,
    y: matrixY,
    elevation: elevation,
  });
};

renderer.domElement.addEventListener(
  "mousemove",
  handleMouseMove
);

        controls.enableDamping = true;

        controls.dampingFactor =
          0.08;

        controls.enablePan = true;

        controls.enableZoom = true;

        controls.minDistance = 80;

        controls.maxDistance = 1000;

        // Center of terrain
        controls.target.set(
          0,
          90,
          0
        );

        // =================================================
        // CAMERA
        // =================================================

        camera.position.set(
          0,
          220,
          330
        );

        camera.lookAt(
          controls.target
        );

        controls.update();

        // =================================================
        // RENDER LOOP
        // =================================================

        let animationFrame;

        const animate = () => {

          animationFrame =
            requestAnimationFrame(
              animate
            );

          controls.update();

          renderer.render(
            scene,
            camera
          );
        };

        animate();

        // =================================================
        // CLEANUP
        // =================================================

        return () => {
          cancelAnimationFrame(
            animationFrame
          );

          renderer.domElement.removeEventListener(
            "mousemove",
            handleMouseMove
          );

          controls.dispose();

          geometry.dispose();

          material.dispose();

          renderer.dispose();

          if (mountRef.current) {
            mountRef.current.removeChild(renderer.domElement);
          }
        };
      }
    );

    // =====================================================
    // RESIZE
    // =====================================================

    const handleResize = () => {

      if (!mount) return;

      camera.aspect =
        mount.clientWidth /
        mount.clientHeight;

      camera.updateProjectionMatrix();

      renderer.setSize(
        mount.clientWidth,
        mount.clientHeight
      );
    };

    window.addEventListener(
      "resize",
      handleResize
    );

    // =====================================================
    // OUTER CLEANUP
    // =====================================================

    return () => {

      window.removeEventListener(
        "resize",
        handleResize
      );

      if (
        mount &&
        mount.contains(
          renderer.domElement
        )
      ) {
        mount.removeChild(
          renderer.domElement
        );
      }

      renderer.dispose();
    };

  }, [heightmapData]);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "650px",
        background: "#111111",
        borderRadius: "12px",
        overflow: "hidden",
      }}
    >

      {/* THREE.JS */}
      <div
        ref={mountRef}
        style={{
          width: "100%",
          height: "100%",
        }}
      />

      {/* =================================================
          COLORBAR
          ================================================= */}

      <div
        style={{
          position: "absolute",
          right: "25px",
          top: "50%",
          transform:
            "translateY(-50%)",
          width: "28px",
          height: "260px",
          background:
            "linear-gradient(to top, blue, cyan, lime, yellow, red)",
          border:
            "1px solid #cccccc",
        }}
      >

        <span
          style={{
            position: "absolute",
            right: "35px",
            bottom: "-5px",
            color: "white",
            fontSize: "12px",
            whiteSpace:
              "nowrap",
          }}
        >
          Low
        </span>

        <span
          style={{
            position: "absolute",
            right: "35px",
            top: "-5px",
            color: "white",
            fontSize: "12px",
            whiteSpace:
              "nowrap",
          }}
        >
          High
        </span>

      </div>

      {/* =================================================
          LABEL
          ================================================= */}

      <div
        style={{
          position: "absolute",
          top: "15px",
          left: "20px",
          color: "white",
          fontSize: "16px",
          fontWeight: "600",
          pointerEvents: "none",
        }}
      >
        SIH 26175 Terrain Reconstruction
      </div>

      {/* =================================================
    HOVER COORDINATES
    ================================================= */}

{hoverInfo && (
  <div
    style={{
      position: "absolute",
      top: "55px",
      left: "20px",
      background: "rgba(0, 0, 0, 0.85)",
      color: "white",
      padding: "10px 14px",
      borderRadius: "8px",
      fontSize: "13px",
      lineHeight: "1.6",
      pointerEvents: "none",
      border: "1px solid #555",
      minWidth: "150px",
      boxShadow:
        "0 4px 12px rgba(0,0,0,0.4)",
    }}
  >
    <div>
      <strong>X:</strong> {hoverInfo.x}
    </div>

    <div>
      <strong>Y:</strong> {hoverInfo.y}
    </div>

    <div>
      <strong>Elevation:</strong>{" "}
      {hoverInfo.elevation.toFixed(2)} m
    </div>
  </div>
)}

      <div
        style={{
          position: "absolute",
          bottom: "15px",
          left: "20px",
          color: "#bbbbbb",
          fontSize: "12px",
          pointerEvents: "none",
        }}
      >
        Drag to rotate • Scroll to zoom • Right-drag to pan
      </div>

    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] =
    useState(null);

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [heightmapUrl, setHeightmapUrl] =
    useState(null);

  const [textureUrl, setTextureUrl] =
    useState(null);

  const [metadataUrl, setMetadataUrl] = useState(null);

  const [heightmapData, setHeightmapData] = useState(null);

  const handleUpload = async (event) => {
    const file =
      event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setLoading(true);

    try {
      const data =
        await uploadImage(file);

      console.log(
        "Backend response:",
        data
      );

      setResult(data);

      if (
        data.success &&
        data.results
      ) {
        const backendUrl =
          "http://127.0.0.1:8000";

        const newHeightmapUrl =
          `${backendUrl}${data.results.heightmap}`;

        const newMetadataUrl =
          `${backendUrl}${data.results.metadata}`;

        const newTextureUrl =
          `${backendUrl}${data.results.texture}`;

        const newHeightmapData = data.results.heightmap_data;

        console.log(
          "Heightmap URL:",
          newHeightmapUrl
        );

        console.log(
          "Metadata URL:",
          newMetadataUrl
        );

        console.log(
          "Texture URL:",
          newTextureUrl
        );

        console.log(
          "Heightmap data:",
          newHeightmapData
        );

        setHeightmapUrl(
          newHeightmapUrl
        );

        setMetadataUrl(
          newMetadataUrl
        );

        setTextureUrl(
          newTextureUrl
        );

        setHeightmapData(newHeightmapData);
      }

    } catch (error) {
      console.error(
        "Upload error:",
        error
      );

      alert(
        "Could not connect to API"
      );
    }

    setLoading(false);
  };


  return (
    <div className="app">

      <header className="header">
        <h1>
          DepthWizard - TerraVision
        </h1>
      </header>


      <main className="main-container">

        <aside className="sidebar">

          <input
            type="file"
            id="imageUpload"
            accept=".jpg,.jpeg,.png,.tif,.tiff"
            style={{
              display: "none",
            }}
            onChange={handleUpload}
          />


          <button
            className="upload-button"
            onClick={() =>
              document
                .getElementById(
                  "imageUpload"
                )
                .click()
            }
          >
            + Upload Image
          </button>


          {selectedFile && (
            <p>
              Selected:{" "}
              {selectedFile.name}
            </p>
          )}


          <section className="panel">

            <h2>
              Image Input Type
            </h2>

            <label>
              <input
                type="radio"
                name="inputType"
              />
              GeoTIFF
            </label>

            <label>
              <input
                type="radio"
                name="inputType"
                defaultChecked
              />
              JPG / PNG
            </label>

          </section>


          <section className="panel">

            <h2>
              Analysis
            </h2>

            <div className="analysis-item">
              <span>
                Height
              </span>

              <strong>
                {loading
                  ? "..."
                  : "-- m"}
              </strong>
            </div>


            <div className="analysis-item">

              <span>
                Slope
              </span>

              <strong>
                {loading
                  ? "..."
                  : "--°"}
              </strong>

            </div>


            <div className="analysis-item">

              <span>
                Coordinates
              </span>

              <strong>
                --
              </strong>

            </div>

          </section>

        </aside>


        <section className="visualization">

          <h2>
            3D Visualization
          </h2>


          <div
            className="visualization-box"
            style={{
              minHeight: "600px",
            }}
          >

            {!selectedFile ? (

              <div className="terrain-placeholder">

                <div className="terrain-grid"></div>

                <span>
                  Upload an image to generate terrain
                </span>

              </div>

            ) : loading ? (

              <div className="terrain-placeholder">

                <div className="terrain-grid"></div>

                <span>
                  Analyzing image...
                </span>

              </div>

            ) : result?.success ? (

              <TerrainViewer
                heightmapUrl={heightmapUrl}
                metadataUrl={metadataUrl}
                heightmapData={heightmapData}
              />

            ) : (

              <div className="terrain-placeholder">

                <span>
                  Processing failed
                </span>

              </div>

            )}

          </div>


          <div className="resolution">

            <span>
              Resolution
            </span>

            <span>
              {result?.results
                ? result.results.run_dir
                : "--"}
            </span>

          </div>

        </section>

      </main>

    </div>
  );
}


export default App;


