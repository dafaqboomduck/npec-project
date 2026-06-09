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
  const containerRef = useRef<HTMLDivElement>(null);
  const [brushSize, setBrushSize] = useState(20);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState<'brush' | 'eraser' | 'pan'>('brush');
  const maskOpacity = 1.0;
  const [isLoaded, setIsLoaded] = useState(false);
  const historyRef = useRef<ImageData[]>([]);
  const [canUndo, setCanUndo] = useState(false);

  // Zoom & Pan States
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [spacePressed, setSpacePressed] = useState(false);

  const resetZoomPan = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  const saveStateToHistory = () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    try {
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      historyRef.current.push(imgData);
      if (historyRef.current.length > 30) {
        historyRef.current.shift();
      }
      setCanUndo(true);
    } catch (err) {
      console.error('Failed to save state to history:', err);
    }
  };

  const handleUndo = () => {
    if (!canvasRef.current || historyRef.current.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const previousState = historyRef.current.pop();
    if (previousState) {
      ctx.putImageData(previousState, 0, 0);
    }
    setCanUndo(historyRef.current.length > 0);
  };

  // Keyboard shortcut Ctrl+Z / Cmd+Z for Undo & Space keydown for panning
  useEffect(() => {
    if (!inline && !isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        handleUndo();
      }
      if (e.code === 'Space' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        setSpacePressed(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setSpacePressed(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [inline, isOpen, canUndo]);

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
        ctx.drawImage(mask, 0, 0, canvas.width, canvas.height); // stretch mask to image size
        setIsLoaded(true);
        historyRef.current = [];
        setCanUndo(false);
        resetZoomPan();
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

  // Mouse & Touch Pan-Zoom Handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    const isMiddleClick = e.button === 1;
    if (tool === 'pan' || spacePressed || isMiddleClick) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    stopDrawing();
  };

  const handleMouseLeave = () => {
    setIsPanning(false);
    stopDrawing();
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (tool === 'pan') {
      const touch = e.touches[0];
      setIsPanning(true);
      setPanStart({ x: touch.clientX - offset.x, y: touch.clientY - offset.y });
    } else {
      startDrawing(e);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isPanning) {
      const touch = e.touches[0];
      setOffset({
        x: touch.clientX - panStart.x,
        y: touch.clientY - panStart.y,
      });
    } else {
      draw(e);
    }
  };

  const handleTouchEnd = () => {
    setIsPanning(false);
    stopDrawing();
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (!containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - containerRect.left;
    const mouseY = e.clientY - containerRect.top;

    const zoomFactor = 1.15;
    let newScale = scale;
    if (e.deltaY < 0) {
      newScale = Math.min(scale * zoomFactor, 10);
    } else {
      newScale = Math.max(scale / zoomFactor, 0.5);
    }

    // Keep point under cursor invariant
    const newOffsetX = mouseX - (mouseX - offset.x) * (newScale / scale);
    const newOffsetY = mouseY - (mouseY - offset.y) * (newScale / scale);

    setScale(newScale);
    setOffset({ x: newOffsetX, y: newOffsetY });
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    const isMiddleClick = 'button' in e && e.button === 1;
    if (tool === 'pan' || spacePressed || isMiddleClick) {
      return;
    }
    saveStateToHistory();
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
    if (!isDrawing || !canvasRef.current || isPanning) return;
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
              <button
                className={`me-toggle-btn ${tool === 'pan' ? 'active' : ''}`}
                onClick={() => setTool('pan')}
              >
                ✋ Pan & Zoom
              </button>
            </div>
          </div>

          <div className="me-tool-group">
            <label>Zoom: {(scale * 100).toFixed(0)}%</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="me-btn me-btn--secondary"
                onClick={() => setScale((prev) => Math.min(prev * 1.2, 10))}
                style={{
                  flex: 1,
                  padding: '6px',
                  fontSize: '0.8rem',
                  margin: 0,
                }}
                title="Zoom In"
              >
                ➕
              </button>
              <button
                className="me-btn me-btn--secondary"
                onClick={() => setScale((prev) => Math.max(prev / 1.2, 0.5))}
                style={{
                  flex: 1,
                  padding: '6px',
                  fontSize: '0.8rem',
                  margin: 0,
                }}
                title="Zoom Out"
              >
                ➖
              </button>
              <button
                className="me-btn me-btn--secondary"
                onClick={resetZoomPan}
                style={{
                  flex: 1.5,
                  padding: '6px',
                  fontSize: '0.8rem',
                  margin: 0,
                }}
                title="Reset Zoom & Pan"
              >
                🔄 Reset
              </button>
            </div>
          </div>

          <div className="me-tool-group">
            <label>History</label>
            <button
              className="me-btn me-btn--secondary"
              onClick={handleUndo}
              disabled={!canUndo}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                padding: '8px',
                fontSize: '0.8rem',
                margin: 0,
              }}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M3 7v6h6" />
                <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
              </svg>
              Undo (Ctrl+Z)
            </button>
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

          <div className="me-help">
            <p>
              💡 Tips:
              <br />• Use <b>Scroll Wheel</b> to zoom.
              <br />• Select <b>Pan</b> tool or hold <b>Spacebar</b> + drag to
              pan.
            </p>
          </div>
        </aside>

        <div
          ref={containerRef}
          className="me-canvas-container"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          style={{
            cursor:
              tool === 'pan' || spacePressed
                ? isPanning
                  ? 'grabbing'
                  : 'grab'
                : 'crosshair',
          }}
        >
          {!isLoaded && (
            <div className="me-loading">Loading high-res scan...</div>
          )}

          <div
            className="me-workspace"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: '0 0',
            }}
          >
            {/* Background image (locked) */}
            <img
              src={originalImageUrl}
              className="me-background-img"
              alt="Scan background"
              draggable={false}
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
    </div>
  );

  if (inline) return editorWindow;

  return <div className="mask-editor-overlay">{editorWindow}</div>;
}
