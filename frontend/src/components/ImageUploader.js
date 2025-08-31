import React, { useState, useEffect, useCallback } from "react";
import axios from "../axiosConfig";

const ImageUploader = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [responseData, setResponseData] = useState(null);
  const [showTracks, setShowTracks] = useState(true); // Show tracks by default
  const [isDragging, setIsDragging] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const userAgent = navigator.userAgent || navigator.vendor || window.opera;
    if (
      /android|iphone|ipad|iPod|opera mini|iemobile|wpdesktop/i.test(
        userAgent.toLowerCase()
      )
    ) {
      setIsMobile(true);
    }
  }, []);

  const resetUploader = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setUploading(false);
    setUploadSuccess(null);
    setUploadError(null);
    setResponseData(null);
    setShowTracks(false);
    setIsDragging(false);
  };

  const processFile = (file) => {
    try {
      if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file.");
        return;
      }

      setSelectedFile(file);
      setUploadSuccess(null);
      setUploadError(null);

      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
    } catch (error) {
      console.error("Error in processFile:", error);
      setUploadError("Failed to process the selected file.");
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select an image to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    setUploading(true);
    setUploadSuccess(null);
    setUploadError(null);

    try {
      const response = await axios.post("/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.status === 200 && response.data) {
        setUploadSuccess("Image matched successfully!");
        setResponseData(response.data);
        setSelectedFile(null);
        setPreviewUrl(null);
      } else {
        throw new Error("Upload failed.");
      }
    } catch (error) {
      console.error("Upload error:", error);
      if (error.response && error.response.data && error.response.data.message) {
        setUploadError(error.response.data.message);
      } else {
        setUploadError("Failed to upload image. Please try again.");
      }
    } finally {
      setUploading(false);
    }
  };

  const handleUploadAnother = () => {
    resetUploader();
  };

  const handleDragStart = useCallback((event) => {
    event.dataTransfer.clearData();
    event.dataTransfer.setData(
      "text/plain",
      event.target.dataset.item || "file"
    );
  }, []);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsDragging(false);

      setUploadSuccess(null);
      setUploadError(null);
      setResponseData(null);
      setShowTracks(false);

      if (
        event.dataTransfer &&
        event.dataTransfer.files &&
        event.dataTransfer.files.length > 0
      ) {
        const file = event.dataTransfer.files[0];
        console.log("File dropped:", file);
        processFile(file);
      } else {
        console.error("No files found in the drop event.");
      }
    },
    [processFile]
  );

  useEffect(() => {
    window.addEventListener("dragstart", handleDragStart);
    window.addEventListener("dragover", handleDragOver);
    window.addEventListener("dragleave", handleDragLeave);
    window.addEventListener("drop", handleDrop);

    return () => {
      window.removeEventListener("dragstart", handleDragStart);
      window.removeEventListener("dragover", handleDragOver);
      window.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("drop", handleDrop);
    };
  }, [handleDragStart, handleDragOver, handleDragLeave, handleDrop]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Button styles
  const buttonStyle = {
    fontSize: "1.2rem",
    padding: "10px 20px",
    gap: "10px",
    textDecoration: "none",
    cursor: "pointer",
  };

  const dragOverlayStyle = {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    zIndex: 9999,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#fff",
    fontSize: "2rem",
    pointerEvents: "none",
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const renderTrack = (track, index) => {
    return (
      <li key={index} style={{ 
        marginBottom: "12px", 
        padding: "0",
        listStyle: "none",
        borderBottom: index < 7 ? "1px solid #e0e0e0" : "none",
        paddingBottom: "12px"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ flex: 1 }}>
            <span style={{ 
              fontWeight: "600", 
              fontSize: "1rem",
              color: "#333"
            }}>
              {track.position}. {track.title}
            </span>
            {track.duration && (
              <span style={{ 
                marginLeft: "10px", 
                color: "#666", 
                fontSize: "0.9rem" 
              }}>
                {track.duration}
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            {track.available_on?.spotify && (
              track.spotify?.url ? (
                <a
                  href={track.spotify.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ 
                    color: "#1db954", 
                    fontSize: "0.85rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
                    textDecoration: "none",
                    fontWeight: 600
                  }}
                >
                  <i className="bi bi-spotify"></i>
                  <span style={{ display: isMobile ? "none" : "inline" }}>Spotify</span>
                </a>
              ) : (
                <span style={{ 
                  color: "#1db954", 
                  fontSize: "0.85rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px"
                }}>
                  <i className="bi bi-spotify"></i>
                  <span style={{ display: isMobile ? "none" : "inline" }}>Spotify</span>
                </span>
              )
            )}
            {track.youtube ? (
              <a 
                href={track.youtube.url || track.youtube.lucky_url} 
                target="_blank" 
                rel="noopener noreferrer"
                style={{ 
                  backgroundColor: "#ff0000",
                  color: "white",
                  padding: "6px 14px",
                  borderRadius: "20px",
                  textDecoration: "none",
                  fontSize: "0.85rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  fontWeight: "500",
                  transition: "all 0.2s",
                  border: "none"
                }}
                onMouseOver={(e) => e.target.style.backgroundColor = "#cc0000"}
                onMouseOut={(e) => e.target.style.backgroundColor = "#ff0000"}
              >
                <i className="bi bi-youtube"></i>
                <span style={{ display: isMobile ? "none" : "inline" }}>
                  Play
                </span>
              </a>
            ) : (
              <span
                style={{ 
                  backgroundColor: "#e0e0e0",
                  color: "#999",
                  padding: "6px 14px",
                  borderRadius: "20px",
                  fontSize: "0.85rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  fontWeight: "500",
                  cursor: "not-allowed"
                }}
              >
                <i className="bi bi-youtube"></i>
                <span style={{ display: isMobile ? "none" : "inline" }}>N/A</span>
              </span>
            )}
          </div>
        </div>
      </li>
    );
  };

  return (
    <div
      className="text-center"
      style={{ position: "relative", paddingBottom: "50px" }}
    >
      {/* Drag Overlay */}
      {isDragging && (
        <div style={dragOverlayStyle}>Drop the image here to upload</div>
      )}

      {/* Upload Section */}
      {!uploadSuccess && <h2>Upload Image</h2>}

      <div className="mb-3" style={{ marginTop: "20px" }}>
        {!uploadSuccess && !selectedFile && (
          <label
            htmlFor="fileInput"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              width: "100%",
              maxWidth: "90%",
              margin: "0 auto",
              padding: "20px",
              backgroundColor: "#007bff",
              color: "#fff",
              textAlign: "center",
              fontSize: "1.2rem",
              borderRadius: "8px",
              cursor: "pointer",
              boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
            }}
          >
            <i
              className="bi bi-camera"
              style={{
                fontSize: "5rem",
                marginBottom: "16px",
              }}
            ></i>
            {isMobile ? "Take Photo" : "Choose File"}
            <input
              id="fileInput"
              type="file"
              accept="image/*"
              capture={isMobile ? "environment" : undefined}
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
          </label>
        )}
      </div>

      {/* Preview Section */}
      <div className="mb-3">
        {previewUrl && (
          <div className="mb-3">
            <img
              src={previewUrl}
              alt="Selected preview"
              style={{
                maxWidth: isMobile ? "90%" : "30%",
                height: "auto",
                borderRadius: "8px",
                boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
              }}
            />
          </div>
        )}

        {selectedFile && !uploadError && (
          <div>
            <button
              className="btn btn-secondary"
              onClick={resetUploader}
              disabled={uploading}
              style={buttonStyle}
            >
              Take Again
            </button>
            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={uploading || !selectedFile}
              style={{ ...buttonStyle, marginLeft: "10px" }}
            >
              {uploading ? "Processing..." : "Confirm"}
            </button>
          </div>
        )}
      </div>

      {/* Feedback Messages */}
      {uploadSuccess && (
        <div className="alert alert-success mt-3" role="alert">
          {uploadSuccess}
        </div>
      )}
      {uploadError && (
        <div className="alert alert-danger mt-3" role="alert">
          {uploadError}
        </div>
      )}

      {uploadError && (
        <div className="mt-3">
          <button
            className="btn btn-secondary"
            onClick={handleUploadAnother}
            style={buttonStyle}
          >
            Upload Another
          </button>
        </div>
      )}

      {/* Results Section */}
      {uploadSuccess && responseData && (
        <div className="mt-4">
          {/* Album Cover */}
          {(responseData.album?.image || responseData.album_image) && (
            <div className="d-flex justify-content-center mb-3">
              <img
                src={responseData.album?.image || responseData.album_image}
                alt="Album cover"
                style={{
                  maxWidth: isMobile ? "70%" : "300px",
                  height: "auto",
                  borderRadius: "8px",
                  boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
                }}
              />
            </div>
          )}

          {/* Album Title */}
          <h3 style={{ margin: "20px 0", color: "#333" }}>
            {responseData.album?.artist || responseData.artist_name} - {responseData.album?.name || responseData.album_name}
          </h3>

          {/* Album Info */}
          <div style={{
            backgroundColor: "#ffffff",
            border: "1px solid #e0e0e0",
            borderRadius: "12px",
            padding: "24px",
            maxWidth: "650px",
            margin: "0 auto 20px",
            textAlign: "left",
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)"
          }}>
            <div style={{ marginBottom: "10px" }}>
              <strong>Release:</strong> {responseData.album?.name || responseData.album_name}
            </div>
            <div style={{ marginBottom: "10px" }}>
              <strong>Artist:</strong> {responseData.album?.artist || responseData.artist_name}
            </div>
            {(responseData.album?.release_date || responseData.release_date) && (
              <div style={{ marginBottom: "10px" }}>
                <strong>Released:</strong> {responseData.album?.release_date || responseData.release_date}
              </div>
            )}
            {(responseData.album?.genres || responseData.genres) && responseData.album?.genres?.length > 0 && (
              <div style={{ marginBottom: "10px" }}>
                <strong>Genres:</strong> {(responseData.album?.genres || responseData.genres).join(", ")}
              </div>
            )}
            {(responseData.price_info?.average_price || responseData.average_price || responseData.median_price || responseData.lowest_price || responseData.market_stats?.num_for_sale || responseData.num_for_sale) && (
              <div style={{ 
                marginBottom: "10px",
                padding: "10px 12px",
                backgroundColor: "#f8fafc",
                border: "1px solid #e6eaf0",
                borderRadius: "8px"
              }}>
                {(() => {
                  const currencySymbols = {
                    USD: "$", EUR: "€", GBP: "£", BRL: "R$", JPY: "¥",
                    AUD: "A$", CAD: "C$", CHF: "CHF", CNY: "¥", SEK: "kr",
                    NOK: "kr", DKK: "kr"
                  };
                  const currency = responseData.price_currency || responseData.market_stats?.currency || responseData.price_info?.currency || "USD";
                  const symbol = currencySymbols[currency] || `${currency} `;
                  const lowest = responseData.lowest_price ?? responseData.market_stats?.lowest_price;
                  const forSale = responseData.num_for_sale ?? responseData.market_stats?.num_for_sale;
                  const median = responseData.median_price ?? responseData.market_stats?.median_price;
                  const avg = responseData.average_price ?? responseData.price_info?.average_price;

                  return (
                    <>
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", color: "#333", marginBottom: 6 }}>
                        {Boolean(lowest) && (
                          <span>
                            <strong>Low:</strong> {symbol}{Number(lowest).toFixed(2)}
                          </span>
                        )}
                        {Boolean(forSale) && (
                          <span>
                            <strong>For sale:</strong> {forSale}
                          </span>
                        )}
                      </div>
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", color: "#333" }}>
                        {Boolean(avg) && (
                          <span>Avg: {symbol}{Number(avg).toFixed(2)}</span>
                        )}
                        {Boolean(median) && (
                          <span>Median: {symbol}{Number(median).toFixed(2)}</span>
                        )}
                      </div>
                    </>
                  );
                })()}
              </div>
            )}
            <div style={{ marginBottom: "10px" }}>
              <strong>confidence:</strong> {Math.round(((responseData.identification?.confidence || responseData.confidence || 0) * 100))}%
            </div>

            {/* Links Section */}
            <div style={{ 
              marginTop: "20px", 
              paddingTop: "20px", 
              borderTop: "1px solid #e0e0e0" 
            }}>
              <h6 style={{ marginBottom: "12px", color: "#666" }}>Links</h6>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                <div>
                  <strong style={{ color: "#666", fontSize: "0.9rem" }}>Discogs:</strong>{" "}
                  {(responseData.links?.discogs || responseData.discogs_url) !== "unavailable" ? (
                    <a 
                      href={responseData.links?.discogs || responseData.discogs_url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{ color: "#0066cc", fontSize: "0.9rem" }}
                    >
                      View on Discogs →
                    </a>
                  ) : (
                    <span style={{ color: "#999", fontSize: "0.9rem" }}>Not available</span>
                  )}
                </div>
                <div>
                  <strong style={{ color: "#666", fontSize: "0.9rem" }}>Spotify:</strong>{" "}
                  {(responseData.links?.spotify || responseData.spotify_url) !== "unavailable" ? (
                    <a 
                      href={responseData.links?.spotify || responseData.spotify_url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{ color: "#1db954", fontSize: "0.9rem" }}
                    >
                      Listen on Spotify →
                    </a>
                  ) : (
                    <span style={{ color: "#999", fontSize: "0.9rem" }}>Not available</span>
                  )}
                </div>
                {Boolean(responseData.links?.bandcamp || responseData.bandcamp_url) && (
                  <div>
                    <strong style={{ color: "#666", fontSize: "0.9rem" }}>Bandcamp:</strong>{" "}
                    <a 
                      href={responseData.links?.bandcamp || responseData.bandcamp_url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{ color: "#1DA0C3", fontSize: "0.9rem" }}
                    >
                      Search on Bandcamp →
                    </a>
                  </div>
                )}
              </div>
            </div>

            {/* Alternatives when confidence is low */}
            {responseData.alternatives && responseData.alternatives.length > 0 && (
              <div style={{ marginTop: "16px", paddingTop: "16px", borderTop: "1px solid #e0e0e0" }}>
                <h6 style={{ color: "#666" }}>Other possible matches</h6>
                <ul style={{ paddingLeft: 18 }}>
                  {responseData.alternatives.map((alt, idx) => (
                    <li key={idx} style={{ marginBottom: 6 }}>
                      <a href={alt.discogs} target="_blank" rel="noopener noreferrer">
                        {alt.artist} - {alt.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Tracks Section - Always Visible */}
            {responseData.tracks?.tracklist && responseData.tracks.tracklist.length > 0 && (
              <div style={{ 
                marginTop: "24px", 
                paddingTop: "24px", 
                borderTop: "1px solid #e0e0e0" 
              }}>
                <h5 style={{ 
                  marginBottom: "16px",
                  fontSize: "1.1rem",
                  color: "#333"
                }}>
                  Tracklist ({responseData.tracks.total || responseData.tracks.tracklist.length} tracks)
                </h5>
                <ul style={{ paddingLeft: 0, margin: 0 }}>
                  {responseData.tracks.tracklist.map((track, index) => renderTrack(track, index))}
                </ul>
              </div>
            )}
          </div>

          {/* Spotify Button */}
          {(responseData.links?.spotify || responseData.spotify_url) && responseData.links?.spotify !== "unavailable" && responseData.spotify_url !== "unavailable" && (
            <div className="mt-3">
              <a
                href={responseData.links?.spotify || responseData.spotify_url}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-success d-flex align-items-center justify-content-center"
                style={{ ...buttonStyle, maxWidth: "250px", margin: "0 auto" }}
              >
                <i className="bi bi-spotify" style={{ marginRight: "10px", fontSize: "1.5rem" }}></i>
                Open in Spotify
              </a>
            </div>
          )}

          {/* Upload Another Button */}
          <div className="mt-4">
            <button
              className="btn btn-secondary"
              onClick={handleUploadAnother}
              style={buttonStyle}
            >
              Upload Another
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;