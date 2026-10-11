/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { HandLandmarkerResult, DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';
import { BaseVisionTask } from '../components/base-vision-task';

// @ts-ignore
import template from '../templates/hand-landmarker.html?raw';
// @ts-ignore

class HandLandmarkerTask extends BaseVisionTask {
  private drawingUtils: DrawingUtils | undefined;

  private numHands = 1;
  // 每200毫秒显示一次坐标，避免Console刷屏
  private lastPalmLogTime = 0;
  private minHandDetectionConfidence = 0.5;
  private minHandPresenceConfidence = 0.5;
  private minTrackingConfidence = 0.5;

  protected override onInitializeUI() {
    // Confidence Sliders
    const setupSlider = (id: string, onChange: (val: number) => void) => {
      const input = document.getElementById(id) as HTMLInputElement;
      const valueDisplay = document.getElementById(`${id}-value`)!;
      if (input && valueDisplay) {
        input.addEventListener('input', () => {
          const val = parseFloat(input.value);
          valueDisplay.innerText = val.toString();
          onChange(val);
        });
      }
    };

    setupSlider('min-hand-detection-confidence', (val) => {
      this.minHandDetectionConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minHandDetectionConfidence: this.minHandDetectionConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-hand-presence-confidence', (val) => {
      this.minHandPresenceConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minHandPresenceConfidence: this.minHandPresenceConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-tracking-confidence', (val) => {
      this.minTrackingConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: this.minTrackingConfidence });
      this.triggerRedetection();
    });

    setupSlider('num-hands', (val) => {
      this.numHands = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', numHands: this.numHands });
      this.triggerRedetection();
    });

    // Custom model options for Hand Landmarker
    this.models = {
      hand_landmarker:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([{ label: 'Hand Landmarker', value: 'hand_landmarker', isDefault: true }]);
    }
  }

  private triggerRedetection() {
    if (this.runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.src) {
        this.detectImage(testImage);
      }
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      numHands: this.numHands,
      minHandDetectionConfidence: this.minHandDetectionConfidence,
      minHandPresenceConfidence: this.minHandPresenceConfidence,
      minTrackingConfidence: this.minTrackingConfidence,
    };
  }

  protected override displayImageResult(result: HandLandmarkerResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    const ctx = imageCanvas.getContext('2d')!;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.beginPath();
    ctx.rect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.clip();

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(ctx);
      else this.drawingUtils = new DrawingUtils(ctx);

      for (const landmarks of result.landmarks) {
        this.drawLandmarks(this.drawingUtils, landmarks);
      }
    }
  }

  protected override displayVideoResult(result: HandLandmarkerResult) {
    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;

    const ctx = this.canvasCtx;
    const width = this.canvasElement.width;
    const height = this.canvasElement.height;

    ctx.clearRect(0, 0, width, height);

    // 保留原来的手部骨架与关键点
    this.drawingUtils = new DrawingUtils(ctx);

    for (const landmarks of result.landmarks) {
      this.drawLandmarks(this.drawingUtils, landmarks);
    }

    // 每帧计算掌心位置
    const palm = this.getPalmPosition(result);

    if (palm) {
      // 将归一化坐标转换为画布上的像素坐标
      const pixelX = palm.x * width;
      const pixelY = palm.y * height;

      // 用一个青色圆点标记掌心
      ctx.save();
      ctx.beginPath();
      ctx.arc(pixelX, pixelY, 10, 0, Math.PI * 2);
      ctx.fillStyle = '#00FFFF';
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#000000';
      ctx.stroke();
      ctx.restore();
    }

    // 只降低日志显示频率；掌心计算仍然每帧执行
    const now = performance.now();

    if (now - this.lastPalmLogTime >= 200) {
      console.log('Palm:', {
        tracked: palm !== null,
        x: palm?.x ?? null,
        y: palm?.y ?? null,
      });

      this.lastPalmLogTime = now;
    }
  }

  private getPalmPosition(result: HandLandmarkerResult) {
    // 取第一只被检测到的手
    const hand = result.landmarks[0];

    // 当前没有检测到手
    if (!hand) {
      return null;
    }

    // 手腕，以及食指、中指、无名指、小指的根部
    const ids = [0, 5, 9, 13, 17];

    const x = ids.reduce((sum, id) => sum + hand[id].x, 0) / ids.length;

    const y = ids.reduce((sum, id) => sum + hand[id].y, 0) / ids.length;

    return { x, y };
  }

  private drawLandmarks(drawingUtils: DrawingUtils, landmarks: any[]) {
    drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: '#00FF00',
      lineWidth: 5,
    });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', lineWidth: 2 });
  }
}

// Singleton instance to support modular cleanup
let activeTask: HandLandmarkerTask | null = null;

export async function setupHandLandmarker(container: HTMLElement) {
  activeTask = new HandLandmarkerTask({
    container,
    template,
    defaultModelName: 'hand_landmarker',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    workerFactory: () =>
      new Worker(new URL('../workers/hand-landmarker.worker.ts', import.meta.url), { type: 'module' }),
  });

  await activeTask.initialize();
}

export function cleanupHandLandmarker() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
