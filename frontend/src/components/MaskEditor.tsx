import React, { useRef, useEffect, useState } from 'react';
import './MaskEditor.css';

interface MaskEditorProps {
  inline?: boolean;
  isOpen: boolean;
  onClose: () => void;
  onSave: (editedMaskBlob: Blob) => void;
  originalImageUrl: string;
  maskImageUrl: string;
  title?: string;
  subtitle?: string;
  saveLabel?: string;
}

export default function MaskEditor({
  inline = false,
  isOpen = false,
  onClose,
  onSave,
  originalImageUrl,
  maskImageUrl,
  title = 'Edit Segmentation Mask',
  subtitle = 'Draw to add roots · Erase to remove noise',
  saveLabel = 'Apply Correction',
}: MaskEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [brushSize, setBrushSize] = useState(20);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush');
  const [maskOpacity, setMaskOpacity] = useState(0.5);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load image and mask onto canvas
  useEffect(() => {
    if ((!inline && !isOpen) || !canvasRef.current) return;

    setIsLoaded(false);

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const img = new Image();
    const mask = new Image();

    img.crossOrigin = 'anonymous';
    mask.crossOrigin = 'anonymous';

    let loadedCount = 0;
    const onLoaded = () => {
      loadedCount++;
      if (loadedCount === 2) {
        // Match canvas size to image resolution
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.fillStyle = '#000000';

        // Draw mask onto canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(mask, 0, 0);
        setIsLoaded(true);
      }
    };

    img.onload = onLoaded;
    mask.onload = onLoaded;
    img.src = originalImageUrl;
    mask.src = maskImageUrl;

    return () => {
      img.onload = null;
      mask.onload = null;
    };
  }, [inline, isOpen, originalImageUrl, maskImageUrl]);

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.beginPath(); // Reset path
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX, clientY;
    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;

    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // if (tool === 'brush') {
    //   ctx.globalCompositeOperation = 'source-over';
    //   ctx.strokeStyle = '#ffffff'; // White for mask pixels
    // } else {
    //   ctx.globalCompositeOperation = 'destination-out';
    //   ctx.strokeStyle = 'rgba(0,0,0,1)';
    // }

    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = tool === 'brush' ? '#ffffff' : '#000000';

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handleSave = () => {
    if (!canvasRef.current) return;

    // Create a new canvas to ensure the mask is exported correctly (white on black/transparent)
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = canvasRef.current.width;
    exportCanvas.height = canvasRef.current.height;
    const eCtx = exportCanvas.getContext('2d');
    if (!eCtx) return;

    // Fill black background (typical for binary masks)
    eCtx.fillStyle = '#000000';
    eCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

    // Draw the current canvas state over it
    eCtx.drawImage(canvasRef.current, 0, 0);

    exportCanvas.toBlob((blob) => {
      if (blob) onSave(blob);
    }, 'image/png');
  };

  if (!inline && !isOpen) return null;

  const editorWindow = (
    <div
      className={`mask-editor-window${inline ? ' mask-editor-window--inline' : ''}`}
    >
      <header className="me-header">
        <div className="me-header-left">
          <h3>{title}</h3>
          <span className="me-subtitle">{subtitle}</span>
        </div>
        <div className="me-header-actions">
          {onClose && (
            <button className="me-btn me-btn--secondary" onClick={onClose}>
              {inline ? 'Skip' : 'Cancel'}
            </button>
          )}
          <button
            className="me-btn me-btn--primary"
            onClick={handleSave}
            disabled={!isLoaded}
          >
            {saveLabel}
          </button>
        </div>
      </header>

      <div className="me-main">
        <aside className="me-sidebar">
          <div className="me-tool-group">
            <label>Tool</label>
            <div className="me-toggle-btns">
              <button
                className={`me-toggle-btn ${tool === 'brush' ? 'active' : ''}`}
                onClick={() => setTool('brush')}
              >
                🖌 Brush
              </button>
              <button
                className={`me-toggle-btn ${tool === 'eraser' ? 'active' : ''}`}
                onClick={() => setTool('eraser')}
              >
                🧽 Eraser
              </button>
            </div>
          </div>

          <div className="me-tool-group">
            <label>Brush Size: {brushSize}px</label>
            <input
              type="range"
              min="1"
              max="100"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
            />
          </div>

          <div className="me-tool-group">
            <label>Mask Opacity: {(maskOpacity * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={maskOpacity}
              onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
            />
          </div>

          <div className="me-help">
            <p>
              💡 Tip: Use the opacity slider to compare the mask against the
              original scan.
            </p>
          </div>
        </aside>

        <div className="me-canvas-container">
          {!isLoaded && (
            <div className="me-loading">Loading high-res scan...</div>
          )}

          {/* Background image (locked) */}
          <img
            src={originalImageUrl}
            className="me-background-img"
            alt="Scan background"
          />

          {/* Drawing Canvas */}
          <canvas
            ref={canvasRef}
            className="me-canvas"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
            style={{ opacity: maskOpacity }}
          />
        </div>
      </div>
    </div>
  );

  if (inline) return editorWindow;

  return <div className="mask-editor-overlay">{editorWindow}</div>;
}
