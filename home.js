(() => {
  'use strict';
  const dialog = document.getElementById('upload-dialog');
  const fileInput = document.getElementById('pdf-upload');
  const dropZone = document.getElementById('drop-zone');
  const error = document.getElementById('upload-error');
  const progress = document.getElementById('upload-progress');
  const close = document.getElementById('close-upload');
  let uploading = false;
  function showError(message) { error.textContent = message || ''; error.hidden = !message; }
  function openUpload() {
    showError(''); fileInput.value = ''; dialog.showModal();
  }
  document.querySelectorAll('[data-upload]').forEach(button => button.addEventListener('click',openUpload));
  close.addEventListener('click',() => dialog.close());
  dialog.addEventListener('cancel',event => { if (uploading) event.preventDefault(); });
  dialog.addEventListener('click',event => {
    const box = dialog.getBoundingClientRect();
    if (!uploading && event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
  });
  async function upload(file) {
    if (uploading || !file) return;
    showError('');
    if (!file.name.toLowerCase().endsWith('.pdf')) return showError('Please choose a PDF resume.');
    if (!file.size || file.size > 6 * 1024 * 1024) return showError('Choose a PDF under 6 MB.');
    uploading = true; fileInput.disabled = true; close.disabled = true;
    dropZone.setAttribute('aria-busy','true'); progress.hidden = false;
    document.getElementById('upload-status').textContent = `Reading ${file.name} and preparing your matches…`;
    try {
      const response = await fetch('/api/resume/pdf',{
        method:'POST',headers:{'Content-Type':'application/pdf','X-Resume-Filename':encodeURIComponent(file.name)},body:file
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'The PDF could not be processed. Please try another copy.');
      window.location.assign('/workspace');
    } catch (problem) {
      showError(problem.message || 'The upload could not finish. Please try again.');
      uploading = false; fileInput.disabled = false; close.disabled = false;
      dropZone.setAttribute('aria-busy','false'); progress.hidden = true; fileInput.value = '';
    }
  }
  fileInput.addEventListener('change',() => upload(fileInput.files[0]));
  ['dragenter','dragover'].forEach(name => dropZone.addEventListener(name,event => {
    event.preventDefault(); if (!uploading) dropZone.classList.add('drag-over');
  }));
  dropZone.addEventListener('dragleave',event => {
    if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop',event => {
    event.preventDefault(); dropZone.classList.remove('drag-over');
    if (event.dataTransfer.files.length !== 1) return showError('Please upload one resume at a time.');
    upload(event.dataTransfer.files[0]);
  });
  // Prevent a missed drop from navigating away from the page with the file.
  window.addEventListener('dragover',event => event.preventDefault());
  window.addEventListener('drop',event => event.preventDefault());
  if (new URL(window.location.href).searchParams.get('upload') === 'resume') openUpload();
})();
