/*
  camera.js - Updated for Render deployment

  - Keeps your existing camera flows for:
    - Student registration
    - Teacher registration
    - Face login

  - NEW: Live liveness overlay during Attendance camera preview
    - Streams frames at ~2-3 fps to /liveness-preview
    - Renders server overlay (LIVE/SPOOF bbox) into <img id="attendanceOverlayImg">
    - Stops preview when you capture the still photo
    - Continues working with Mark Attendance (which already returns overlay too)
    - Added error handling for network issues and model availability

  Attendance page expected element IDs (ensure these exist in attendance.html):
    - Buttons: startCameraAttendance, captureImageAttendance, retakeImageAttendance, markAttendanceBtn
    - Media:   videoAttendance (video), canvasAttendance (canvas), attendanceOverlayImg (img)
    - Status:  attendanceStatus (div/span for messages)
    - Fields:  program, semester, course, student_id (optional, server may read from session)
*/

document.addEventListener('DOMContentLoaded', function () {
  // Enhanced error handling for network requests
  async function makeRequest(url, options = {}) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Request timed out. Please check your internet connection.');
      }
      throw error;
    }
  }

  // Show loading indicator
  function showLoading(element, message = 'Processing...') {
    if (element) {
      element.innerHTML = `
        <div class="d-flex align-items-center">
          <div class="spinner-border spinner-border-sm me-2" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
          ${message}
        </div>
      `;
    }
  }

  // Hide loading indicator
  function hideLoading(element, message = '') {
    if (element) {
      element.innerHTML = message;
    }
  }

  // Enhanced camera access with better error handling
  async function getCameraStream(constraints = {}) {
    const defaultConstraints = {
      video: {
        width: { ideal: 640, max: 1280 },
        height: { ideal: 480, max: 720 },
        facingMode: 'user',
        frameRate: { ideal: 30, max: 60 }
      }
    };

    const finalConstraints = {
      ...defaultConstraints,
      ...constraints
    };

    try {
      // Try with ideal constraints first
      return await navigator.mediaDevices.getUserMedia(finalConstraints);
    } catch (error) {
      console.warn('Failed with ideal constraints, trying basic:', error);
      try {
        // Fallback to basic constraints
        return await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' }
        });
      } catch (fallbackError) {
        console.error('Camera access failed completely:', fallbackError);
        throw new Error('Unable to access camera. Please ensure camera permissions are granted and no other application is using the camera.');
      }
    }
  }

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

    // Start camera with enhanced error handling
    startCameraBtn.addEventListener('click', async function () {
      try {
        startCameraBtn.disabled = true;
        showLoading(startCameraBtn, 'Starting camera...');

        stream = await getCameraStream({
          video: {
            width: { ideal: 400, max: 640 },
            height: { ideal: 300, max: 480 },
            facingMode: 'user'
          }
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
        alert(err.message || 'Could not access the camera. Please make sure it is connected and permissions are granted.');
      } finally {
        startCameraBtn.disabled = false;
        hideLoading(startCameraBtn, 'Start Camera');
      }
    });

    // Capture image
    captureImageBtn.addEventListener('click', function () {
      try {
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
      } catch (err) {
        console.error('Error capturing image:', err);
        alert('Failed to capture image. Please try again.');
      }
    });

    // Retake image
    retakeImageBtn.addEventListener('click', async function () {
      try {
        retakeImageBtn.disabled = true;
        showLoading(retakeImageBtn, 'Restarting...');

        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);
        faceImageInput.value = '';

        stream = await getCameraStream({
          video: {
            width: { ideal: 400, max: 640 },
            height: { ideal: 300, max: 480 },
            facingMode: 'user'
          }
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
        alert(err.message || 'Error restarting camera.');
      } finally {
        retakeImageBtn.disabled = false;
        hideLoading(retakeImageBtn, 'Retake');
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
    let consecutiveErrors = 0;
    const maxConsecutiveErrors = 5;
    const previewCanvas = document.createElement('canvas');
    const previewCtx = previewCanvas.getContext('2d');

    if (!video || !canvas || !startBtn || !captureBtn || !retakeBtn || !markBtn) {
      return;
    }

    function setStatus(msg, isError = false) {
      if (!statusEl) return;
      statusEl.textContent = msg || '';
      statusEl.classList.remove('text-success', 'text-danger', 'text-warning');
      if (isError) {
        statusEl.classList.add('text-danger');
      } else if (msg.includes('SPOOF')) {
        statusEl.classList.add('text-warning');
      } else {
        statusEl.classList.add('text-success');
      }
    }

    function clearOverlay() {
      if (overlayImg) {
        overlayImg.src = '';
        overlayImg.classList.add('d-none');
      }
    }

    async function startCamera() {
      try {
        startBtn.disabled = true;
        showLoading(startBtn, 'Starting camera...');

        stream = await getCameraStream({
          video: {
            width: { ideal: 640, max: 1280 },
            height: { ideal: 480, max: 720 },
            facingMode: 'user'
          }
        });

        video.srcObject = stream;
        await video.play();

        startBtn.classList.add('d-none');
        captureBtn.classList.remove('d-none');
        retakeBtn.classList.add('d-none');
        markBtn.disabled = true;

        video.classList.remove('d-none');
        canvas.classList.add('d-none');
        setStatus('Camera started. Live preview will begin shortly...');
        clearOverlay();

        // Configure preview canvas size (lower res for network efficiency)
        previewCanvas.width = 480;
        previewCanvas.height = Math.round(previewCanvas.width * (video.videoHeight || 480) / (video.videoWidth || 640));

        // Wait a moment for video to stabilize before starting preview
        setTimeout(() => {
          if (stream && video.readyState >= 2) {
            startPreview();
          }
        }, 1000);

      } catch (err) {
        console.error('Error accessing camera:', err);
        alert(err.message || 'Could not access the camera. Please ensure permissions are granted.');
      } finally {
        startBtn.disabled = false;
        hideLoading(startBtn, 'Start Camera');
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
      consecutiveErrors = 0;

      // Stream frames at ~2-3 fps to reduce server load and handle Render's limitations
      const intervalMs = 500; // 2 fps for better stability on cloud hosting
      previewTimer = setInterval(async () => {
        if (!previewActive || previewBusy || !stream || video.readyState < 2) return;
        
        previewBusy = true;
        try {
          // Draw current frame to offscreen canvas
          previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
          const frameDataUrl = previewCanvas.toDataURL('image/jpeg', 0.5); // Lower quality for faster transfer

          const data = await makeRequest('/liveness-preview', {
            method: 'POST',
            body: JSON.stringify({ face_image: frameDataUrl })
          });

          // Reset error counter on success
          consecutiveErrors = 0;

          if (overlayImg && data.overlay) {
            overlayImg.src = data.overlay;
            overlayImg.classList.remove('d-none');
          }

          // Enhanced status feedback
          if (typeof data.live === 'boolean' && typeof data.live_prob === 'number') {
            const confidenceLevel = data.live_prob >= 0.9 ? 'High' : data.live_prob >= 0.7 ? 'Good' : 'Low';
            setStatus(`${data.live ? 'LIVE' : 'SPOOF'} (${confidenceLevel}: ${data.live_prob.toFixed(2)})`, !data.live);
          } else if (data.message) {
            setStatus(data.message, !data.success);
          }

        } catch (error) {
          consecutiveErrors++;
          console.warn(`Preview failed (${consecutiveErrors}/${maxConsecutiveErrors}):`, error);
          
          if (consecutiveErrors >= maxConsecutiveErrors) {
            setStatus('Preview temporarily unavailable. You can still capture and mark attendance.', true);
            stopPreview(); // Stop trying after too many failures
          } else if (consecutiveErrors === 1) {
            setStatus('Preview connection issue...', true);
          }
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
      consecutiveErrors = 0;
    }

    function captureFrame() {
      // Stop live preview while capturing a still
      stopPreview();
      setStatus('Image captured. Ready to mark attendance.');

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
      try {
        retakeBtn.disabled = true;
        showLoading(retakeBtn, 'Restarting...');

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        capturedDataUrl = '';
        clearOverlay();
        await startCamera();
      } catch (err) {
        console.error('Error during retake:', err);
        alert(err.message || 'Error restarting camera.');
      } finally {
        retakeBtn.disabled = false;
        hideLoading(retakeBtn, 'Retake');
      }
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
        showLoading(markBtn, 'Marking attendance...');
        setStatus('Processing attendance... Please wait.');

        const data = await makeRequest('/mark-attendance', {
          method: 'POST',
          body: JSON.stringify(payload)
        });

        // Show final overlay image with LIVE/SPOOF bbox (from server)
        if (overlayImg && data.overlay) {
          overlayImg.src = data.overlay;
          overlayImg.classList.remove('d-none');
        }

        if (data.success) {
          setStatus(data.message || 'Attendance marked successfully.', false);
          // Auto-refresh after successful attendance
          setTimeout(() => {
            window.location.reload();
          }, 3000);
        } else {
          setStatus(data.message || 'Failed to mark attendance.', true);
        }

      } catch (err) {
        console.error('markAttendance error:', err);
        const errorMsg = err.message.includes('timed out') 
          ? 'Request timed out. Please check your connection and try again.'
          : err.message.includes('models not available')
          ? 'Face recognition service is temporarily unavailable. Please try again later.'
          : 'An error occurred while marking attendance. Please try again.';
        setStatus(errorMsg, true);
      } finally {
        markBtn.disabled = false;
        hideLoading(markBtn, 'Mark Attendance');
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

    // Handle visibility change (pause preview when tab not active)
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && previewActive) {
        stopPreview();
      } else if (!document.hidden && stream && video.readyState >= 2 && !previewActive) {
        setTimeout(startPreview, 500);
      }
    });
  }

  // Enhanced auto face login with better error handling
  function setupAutoFaceLogin() {
    const autoLoginBtn = document.getElementById('autoFaceLoginBtn');
    const roleSelect = document.getElementById('faceRole');
    
    if (!autoLoginBtn) return;

    autoLoginBtn.addEventListener('click', async function() {
      try {
        autoLoginBtn.disabled = true;
        showLoading(autoLoginBtn, 'Accessing camera...');

        const role = roleSelect ? roleSelect.value : 'student';
        
        const stream = await getCameraStream({
          video: {
            width: { ideal: 640, max: 1280 },
            height: { ideal: 480, max: 720 },
            facingMode: 'user'
          }
        });

        // Create temporary video element for auto-login
        const tempVideo = document.createElement('video');
        tempVideo.srcObject = stream;
        tempVideo.muted = true;
        await tempVideo.play();

        // Wait for video to stabilize
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Capture frame
        const tempCanvas = document.createElement('canvas');
        const ctx = tempCanvas.getContext('2d');
        tempCanvas.width = tempVideo.videoWidth || 640;
        tempCanvas.height = tempVideo.videoHeight || 480;
        ctx.drawImage(tempVideo, 0, 0, tempCanvas.width, tempCanvas.height);
        
        const imageDataURL = tempCanvas.toDataURL('image/jpeg', 0.8);

        // Clean up camera
        stream.getTracks().forEach(track => track.stop());

        showLoading(autoLoginBtn, 'Recognizing face...');

        const data = await makeRequest('/auto-face-login', {
          method: 'POST',
          body: JSON.stringify({
            face_image: imageDataURL,
            face_role: role
          })
        });

        if (data.success) {
          setStatus && setStatus('Login successful! Redirecting...', false);
          setTimeout(() => {
            window.location.href = data.redirect_url;
          }, 1000);
        } else {
          alert(data.message || 'Face recognition failed. Please try again.');
        }

      } catch (err) {
        console.error('Auto face login error:', err);
        alert(err.message || 'Auto face login failed. Please try manual login.');
      } finally {
        autoLoginBtn.disabled = false;
        hideLoading(autoLoginBtn, 'Auto Face Login');
      }
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

  // Attendance Camera (with live liveness preview)
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
    studentIdInputId: 'student_id',
  });

  // Setup auto face login if available
  setupAutoFaceLogin();

  // Global error handling for network issues
  window.addEventListener('online', () => {
    console.log('Connection restored');
    setStatus && setStatus('Connection restored', false);
  });

  window.addEventListener('offline', () => {
    console.log('Connection lost');
    setStatus && setStatus('No internet connection', true);
  });
});