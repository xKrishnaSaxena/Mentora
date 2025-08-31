import React from "react";

export default function AudioPlayer({ base64, className = "" }) {
  if (!base64) return null;
  const src = `data:audio/mp3;base64,${base64}`;
  return (
    <audio className={className} controls src={src}>
      Your browser does not support the audio element.
    </audio>
  );
}
