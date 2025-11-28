import { useState } from "react";
import "./Analizador.css";
import { FaMicrophone, FaVideo, FaFileAlt, FaCloudUploadAlt } from "react-icons/fa";

function Analizador() {
  const [file, setFile] = useState(null);
  const [resultado, setResultado] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mensaje, setMensaje] = useState("");

  const handleFileSelect = (e) => {
    const archivo = e.target.files[0];
    setFile(archivo);
    setMensaje(`Archivo seleccionado: ${archivo.name}`);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setMensaje("⚠️ Selecciona un archivo antes de subir.");
      return;
    }

    setLoading(true);
    setMensaje("⏳ Subiendo y analizando el archivo...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:4000/analizar", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResultado(data);
      setMensaje("✅ Archivo analizado correctamente.");
    } catch (error) {
      setMensaje("❌ Error al conectar con el servidor.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analizador-container">
      <div className="header">
        <img src="/amana-logo.png" alt="Logo" className="logo" />
      </div>

      <h2 className="titulo">REGISTRO Y CARGA DE MATERIAL</h2>

      <div
        className="dropzone"
        onClick={() => document.getElementById("file-input").click()}
      >
        <FaCloudUploadAlt size={50} color="#ff7b00" />
        <p>
          <strong>Arrastra y suelta los archivos</strong>
          <br />
          o selecciona para cargarlos
        </p>
        <input
          type="file"
          id="file-input"
          accept="audio/*,video/*"
          onChange={handleFileSelect}
          hidden
        />
      </div>

      {file && (
        <p className="archivo-seleccionado">
          📁 <strong>{file.name}</strong>
        </p>
      )}

      {mensaje && <p className="mensaje">{mensaje}</p>}

      <div className="botones">
        <button className="btn naranja">
          <FaMicrophone /> Grabación de audio
        </button>
        <button className="btn naranja">
          <FaVideo /> Grabación de video
        </button>
        <button className="btn naranja">
          <FaFileAlt /> Entrevista estructurada
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <button type="submit" className="btn-principal" disabled={loading}>
          {loading ? "Analizando..." : "Subir"}
        </button>
      </form>

      {loading && <div className="spinner"></div>}

      {resultado && (
  <div className="resultado">
    <h3>Resultado del análisis</h3>

    <p>
      <strong>Emoción predominante:</strong>{" "}
      {resultado.emocion_predominante}
    </p>
    <p>
      <strong>Porcentaje:</strong> {resultado.porcentaje}
    </p>

    <h4>Distribución de emociones:</h4>
    <pre>{JSON.stringify(resultado.distribucion, null, 2)}</pre>

    
    {resultado.imagen_emocion && (
      <div className="imagen-resultado">
        <h4>Frame representativo:</h4>
        <img
          src={`data:image/jpeg;base64,${resultado.imagen_emocion}`}
          alt="Frame con emoción dominante"
          className="imagen-emocion"
        />
      </div>
    )}
  </div>
)}

    </div>
  );
}

export default Analizador;
