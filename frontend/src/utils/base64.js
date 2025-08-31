export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      resolve(result.split(",")[1] || result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
