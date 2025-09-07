/*
  camera.js

  - Keeps your existing camera flows for:
    - Student registration
    - Teacher registration
    - Face login

  - NEW: Live liveness overlay during Attendance camera preview
    - Streams frames at ~2-3 fps to /liveness-preview
    - Renders server overlay (LIVE/SPOOF bbox) into <img id="attendanceOverlayImg">
    - Stops preview when you capture the still photo
    - Continues working with Mark Attendance (which already returns overlay too)

  Attendance page expected element IDs (ensure these exist in attendance.html):
    - Buttons: startCameraAttendance, captureImageAttendance, retakeImageAttendance, markAttendanceBtn
    - Media:   videoAttendance (video), canvasAttendance (canvas), attendanceOverlayImg (img)
    - Status:  attendanceStatus (div/span for messages)
    - Fields:  program, semester, course, student_id (optional, server may read from session)
*/

document.addEventListener('DOMContentLoaded', function () {
  // Reusable section setup for Registration/Login
  function setupCameraSection(config) {
    const video = document.getElementById(config.videoId);
    const canvas = document.getElementById(config.canvasId);
    const startCameraBtn = document.getElementById(config.startCameraBtnId);
    const captureImageBtn = document.getElementById(config.captureImageBtnId);
    const retakeImageBtn = document.getElementById(config.retakeImageBtnId);
    const cameraOverlay = document.getElementById(config.cameraOverlayId);
    const faceImageInput = document.getElementById(config.faceImageInputId);
    const actionBtn = document.getElementById(config.actionBtnId);

    let stream = null;

    if (
      !video ||
      !canvas ||
      !startCameraBtn ||
      !captureImageBtn ||
      !retakeImageBtn ||
      !cameraOverlay ||
      !faceImageInput ||
      !actionBtn
    ) {
      // Missing elements → skip this section gracefully
      return;
    }

    // Start camera
    startCameraBtn.addEventListener('click', async function () {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 400,
            height: 300,
            facingMode: 'user',
          },
        });
        video.srcObject = stream;
        await video.play();

        startCameraBtn.classList.add('d-none');
        captureImageBtn.classList.remove('d-none');
        retakeImageBtn.classList.add('d-none');
        cameraOverlay.classList.add('d-none');
        video.classList.remove('d-none');
        canvas.classList.add('d-none');
        actionBtn.disabled = true;
      } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Could not access the camera. Please make sure it is connected and permissions are granted.');
      }
    });

    // Capture image
    captureImageBtn.addEventListener('click', function () {
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth || 400;
      canvas.height = video.videoHeight || 300;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageDataURL = canvas.toDataURL('image/jpeg', 0.8);
      faceImageInput.value = imageDataURL;

      cameraOverlay.classList.remove('d-none');
      captureImageBtn.classList.add('d-none');
      retakeImageBtn.classList.remove('d-none');
      video.classList.add('d-none');
      canvas.classList.remove('d-none');
      actionBtn.disabled = false;

      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
    });

    // Retake image
    retakeImageBtn.addEventListener('click', async function () {
      try {
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);
        faceImageInput.value = '';

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 400,
            height: 300,
            facingMode: 'user',
          },
        });
        video.srcObject = stream;
        await video.play();

        cameraOverlay.classList.add('d-none');
        captureImageBtn.classList.remove('d-none');
        retakeImageBtn.classList.add('d-none');
        video.classList.remove('d-none');
        canvas.classList.add('d-none');
        actionBtn.disabled = true;
      } catch (err) {
        console.error('Error restarting camera:', err);
      }
    });
  }

  // Attendance-specific section with LIVE overlay streaming
  function setupAttendanceSection(config) {
    const video = document.getElementById(config.videoId);
    const canvas = document.getElementById(config.canvasId);
    const startBtn = document.getElementById(config.startCameraBtnId);
    const captureBtn = document.getElementById(config.captureImageBtnId);
    const retakeBtn = document.getElementById(config.retakeImageBtnId);
    const markBtn = document.getElementById(config.markBtnId);
    const overlayImg = document.getElementById(config.overlayImgId);
    const statusEl = document.getElementById(config.statusId);

    // Optional fields (server may use session for student_id)
    const programEl = document.getElementById(config.programId);
    const semesterEl = document.getElementById(config.semesterId);
    const courseEl = document.getElementById(config.courseId);
    const studentIdEl = document.getElementById(config.studentIdInputId);

    let stream = null;
    let capturedDataUrl = '';

    // Live preview control
    let previewActive = false;
    let previewBusy = false;
    let previewTimer = null;
    const previewCanvas = document.createElement('canvas');
    const previewCtx = previewCanvas.getContext('2d');

    if (!video || !canvas || !startBtn || !captureBtn || !retakeBtn || !markBtn) {
      return;
    }

    function setStatus(msg, isError = false) {
      if (!statusEl) return;
      statusEl.textContent = msg || '';
      statusEl.classList.remove('text-success', 'text-danger');
      statusEl.classList.add(isError ? 'text-danger' : 'text-success');
    }

    function clearOverlay() {
      if (overlayImg) {
        overlayImg.src = '';
        overlayImg.classList.remove('d-none');
      }
    }

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });
        video.srcObject = stream;
        await video.play();

        startBtn.classList.add('d-none');
        captureBtn.classList.remove('d-none');
        retakeBtn.classList.add('d-none');
        markBtn.disabled = true;

        video.classList.remove('d-none');
        canvas.classList.add('d-none');
        setStatus('');
        clearOverlay();

        // Configure preview canvas size (lower res for network)
        previewCanvas.width = 480;
        previewCanvas.height = Math.round(previewCanvas.width * (video.videoHeight || 480) / (video.videoWidth || 640));

        startPreview();
      } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Could not access the camera. Please ensure permissions are granted.');
      }
    }

    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
    }

    function startPreview() {
      if (previewActive) return;
      previewActive = true;

      // Stream frames at ~2-3 fps to reduce server load
      const intervalMs = 400; // 2.5 fps
      previewTimer = setInterval(async () => {
        if (!previewActive || previewBusy || !stream) return;
        previewBusy = true;
        try {
          // Draw current frame to offscreen canvas
          previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
          const frameDataUrl = previewCanvas.toDataURL('image/jpeg', 0.6);

          // Minimal payload (no form fields needed for preview)
          const res = await fetch('/liveness-preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ face_image: frameDataUrl }),
          });
          const data = await res.json();

          if (overlayImg && data.overlay) {
            overlayImg.src = data.overlay;
            overlayImg.classList.remove('d-none');
          }

          // Optional: status text based on live/spoof
          if (typeof data.live === 'boolean' && typeof data.live_prob === 'number') {
            setStatus(`${data.live ? 'LIVE' : 'SPOOF'} p=${data.live_prob.toFixed(2)}`, !data.live);
          } else if (data.message) {
            setStatus(data.message, !data.success);
          }
        } catch (e) {
          console.warn('Preview failed:', e);
        } finally {
          previewBusy = false;
        }
      }, intervalMs);
    }

    function stopPreview() {
      previewActive = false;
      if (previewTimer) {
        clearInterval(previewTimer);
        previewTimer = null;
      }
      previewBusy = false;
    }

    function captureFrame() {
      // Stop live preview while capturing a still
      stopPreview();

      const ctx = canvas.getContext('2d');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      capturedDataUrl = canvas.toDataURL('image/jpeg', 0.9);

      captureBtn.classList.add('d-none');
      retakeBtn.classList.remove('d-none');
      markBtn.disabled = false;

      video.classList.add('d-none');
      canvas.classList.remove('d-none');

      stopCamera();
    }

    async function retake() {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      capturedDataUrl = '';
      clearOverlay();
      await startCamera();
    }

    async function markAttendance() {
      try {
        if (!capturedDataUrl) {
          setStatus('Please capture an image first.', true);
          return;
        }

        const payload = {
          student_id:
            (studentIdEl && studentIdEl.value) ||
            (markBtn && markBtn.dataset && markBtn.dataset.studentId) ||
            null,
          program: (programEl && programEl.value) || '',
          semester: (semesterEl && semesterEl.value) || '',
          course: (courseEl && courseEl.value) || '',
          face_image: capturedDataUrl,
        };

        if (!payload.program || !payload.semester || !payload.course) {
          setStatus('Program, Semester, and Course are required.', true);
          return;
        }

        markBtn.disabled = true;
        setStatus('Marking attendance...');

        const res = await fetch('/mark-attendance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        const data = await res.json();

        // Show final overlay image with LIVE/SPOOF bbox (from server)
        if (overlayImg && data.overlay) {
          overlayImg.src = data.overlay;
          overlayImg.classList.remove('d-none');
        }

        if (data.success) {
          setStatus(data.message || 'Attendance marked successfully.', false);
        } else {
          setStatus(data.message || 'Failed to mark attendance.', true);
        }
      } catch (err) {
        console.error('markAttendance error:', err);
        setStatus('An error occurred while marking attendance.', true);
      } finally {
        markBtn.disabled = false;
      }
    }

    // Wire events
    startBtn.addEventListener('click', startCamera);
    captureBtn.addEventListener('click', captureFrame);
    retakeBtn.addEventListener('click', retake);
    markBtn.addEventListener('click', markAttendance);

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
      stopPreview();
      stopCamera();
    });
  }

  // Student Registration Camera
  setupCameraSection({
    videoId: 'videoStudent',
    canvasId: 'canvasStudent',
    startCameraBtnId: 'startCameraStudent',
    captureImageBtnId: 'captureImageStudent',
    retakeImageBtnId: 'retakeImageStudent',
    cameraOverlayId: 'cameraOverlayStudent',
    faceImageInputId: 'face_image_student',
    actionBtnId: 'registerBtnStudent',
  });

  // Teacher Registration Camera
  setupCameraSection({
    videoId: 'videoTeacher',
    canvasId: 'canvasTeacher',
    startCameraBtnId: 'startCameraTeacher',
    captureImageBtnId: 'captureImageTeacher',
    retakeImageBtnId: 'retakeImageTeacher',
    cameraOverlayId: 'cameraOverlayTeacher',
    faceImageInputId: 'face_image_teacher',
    actionBtnId: 'registerBtnTeacher',
  });

  // Face Login Camera (if present)
  setupCameraSection({
    videoId: 'video',
    canvasId: 'canvas',
    startCameraBtnId: 'startCamera',
    captureImageBtnId: 'captureImage',
    retakeImageBtnId: 'retakeImage',
    cameraOverlayId: 'cameraOverlay',
    faceImageInputId: 'face_image',
    actionBtnId: 'faceLoginBtn',
  });

  // Attendance Camera (NEW) — requires the attendance page to include these IDs
  setupAttendanceSection({
    videoId: 'videoAttendance',
    canvasId: 'canvasAttendance',
    startCameraBtnId: 'startCameraAttendance',
    captureImageBtnId: 'captureImageAttendance',
    retakeImageBtnId: 'retakeImageAttendance',
    markBtnId: 'markAttendanceBtn',
    overlayImgId: 'attendanceOverlayImg',
    statusId: 'attendanceStatus',
    // Optional form fields
    programId: 'program',
    semesterId: 'semester',
    courseId: 'course',
    studentIdInputId: 'student_id', // optional; server may use session
  });
});