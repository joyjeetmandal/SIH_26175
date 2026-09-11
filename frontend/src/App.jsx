import React, { useState } from "react";
import { uploadImage } from "./api";
import "./style.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [heightmapUrl, setHeightmapUrl] = useState(null);
  const [textureUrl, setTextureUrl] = useState(null);

  const handleUpload = async (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setLoading(true);

    try {
      const data = await uploadImage(file);

      console.log("Backend response:", data);

      setResult(data);

      if (data.success && data.results) {
      const backendUrl = "http://127.0.0.1:8000";

      setHeightmapUrl(
      `${backendUrl}${data.results.heightmap}`
      );

      setTextureUrl(
        `${backendUrl}${data.results.texture}`
        );
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Could not connect to API");
    }

    setLoading(false);
  };

  return (
    <div className="app">

      <header className="header">
        <h1>DepthWizard - TerraVision</h1>
      </header>

      <main className="main-container">

        <aside className="sidebar">

          {/* Hidden file input */}
          <input
            type="file"
            id="imageUpload"
            accept=".jpg,.jpeg,.png,.tif,.tiff"
            style={{ display: "none" }}
            onChange={handleUpload}
          />

          {/* Upload button */}
          <button
            className="upload-button"
            onClick={() =>
              document.getElementById("imageUpload").click()
            }
          >
            + Upload Image
          </button>

          {/* Show selected file */}
          {selectedFile && (
            <p>
              Selected: {selectedFile.name}
            </p>
          )}

          <section className="panel">
            <h2>Image Input Type</h2>

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
            <h2>Analysis</h2>

            <div className="analysis-item">
              <span>Height</span>

              <strong>
                {loading
                  ? "..."
                  : "-- m"}
              </strong>
            </div>

            <div className="analysis-item">
              <span>Slope</span>

                <strong>
                {loading
                  ? "..."
                  : "--°"}
                </strong>
              </div>

              <div className="analysis-item">
                <span>Coordinates</span>

                <strong>
                   --
                </strong>
              </div>
          </section>

        </aside>

        <section className="visualization">

          <h2>3D Visualization</h2>

         <div className="visualization-box">

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

            <div className="results-container">

              <div className="result-image">
                <h3>Generated Heightmap</h3>

                {heightmapUrl && (
                  <img
                    src={heightmapUrl}
                    alt="Generated heightmap"
                  />
                 )}
                </div>

                <div className="result-image">
                  <h3>Texture</h3>

                  {textureUrl && (
                    <img
                      src={textureUrl}
                      alt="Generated terrain texture"
                    />
                  )}
                </div>

            </div>

          ) : (

            <div className="terrain-placeholder">
              <span>
                Processing failed
              </span>
            </div>

          )}

        </div>

          <div className="resolution">

            <span>Resolution</span>

            <span>
              {result
                ? result.resolution
                : "--"}
            </span>

          </div>

        </section>

      </main>

    </div>
  );
}

export default App;




