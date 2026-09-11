import React, { useState } from 'react';
import DisasterSimulation from './DisasterSimulation'; // The 3D canvas component we built earlier

export default function ThreeJSBridge() {
  const [loading, setLoading] = useState(false);
  const [simulationData, setSimulationData] = useState(null);
  const [localImageUrl, setLocalImageUrl] = useState(null);
  const [error, setError] = useState(null);

  //  1. The Network Function that talks directly to your FastAPI backend
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setSimulationData(null);

    // Create a local URL so Three.js can display the photo skin immediately
    setLocalImageUrl(URL.createObjectURL(file));

    // Package the raw 2D image binary for our backend pipeline upload
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log(" Sending drone image to FastAPI backend...");
      const response = await fetch('http://127.0.0', {
        method: 'POST',
        body: formData, // Sends the file payload to your backend server
      });

      if (!response.ok) {
        throw new Error(`Backend Error: Status code ${response.status}`);
      }

      // 2. Receive the aspect-ratio metadata and coordinate matrix arrays
      const data = await response.json();
      console.log("Data successfully retrieved from backend:", data);
      
      setSimulationData(data);
    } catch (err) {
      console.error("Integration Failure:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', background: '#1c1c1f', color: '#fff', borderRadius: '8px' }}>
      <h2> DepthWizard Control Room Bridge</h2>
      <p style={{ color: '#aaa' }}>Upload a drone image to activate the pretrained AI depth estimation pipeline.</p>
      
      {/* File Input Selector for the Dashboard */}
      <div style={{ marginBottom: '20px' }}>
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload} 
          disabled={loading}
          style={{ background: '#29292e', padding: '10px', borderRadius: '4px', border: '1px solid #444', cursor: 'pointer' }}
        />
      </div>

      {/* Visual State Indicator Managers */}
      {loading && (
        <div style={{ padding: '40px', textAlign: 'center', color: '#3b82f6' }}>
          Depth Engine Active: Extracting terrain matrices and pixel aspects...
        </div>
      )}

      {error && (
        <div style={{ padding: '15px', background: '#7f1d1d', color: '#fca5a5', borderRadius: '4px' }}>
          Connection Failure: {error}. (Is your Python server running via Anaconda Prompt?)
        </div>
      )}

      {/* 🌌 3. The Handshake: Passing your custom metadata directly into the 3D Canvas */}
      {simulationData && !loading && (
        <div>
          <div style={{ display: 'flex', gap: '20px', marginBottom: '15px', background: '#27272a', padding: '12px', borderRadius: '4px', fontSize: '14px' }}>
            <div><strong>Original Size:</strong> {simulationData.meta.original_dimensions.width}px × {simulationData.meta.original_dimensions.height}px</div>
            <div><strong>Aspect Ratio:</strong> {simulationData.meta.aspect_ratio}</div>
            <div><strong>Three.js Grid Resolution:</strong> {simulationData.meta.grid_dimensions.width} × {simulationData.meta.grid_dimensions.height}</div>
            <div><strong>Total Vertices:</strong> {simulationData.total_points}</div>
          </div>

          {/* Renders the automated camera flythrough container view */}
          <DisasterSimulation 
            backendPoints={simulationData} 
            originalImageUrl={localImageUrl} 
            //  NEW: Dynamically feed custom resolution definitions down to Three.js parameters
            gridWidth={simulationData.meta.grid_dimensions.width}
            gridHeight={simulationData.meta.grid_dimensions.height}
          />
        </div>
      )}
    </div>
  );
}