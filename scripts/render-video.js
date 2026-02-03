#!/usr/bin/env node

/**
 * KRACKER Remotion Render Script
 *
 * 프로젝트 폴더의 에셋을 public/output으로 복사하고
 * Remotion 렌더링 후 결과물을 프로젝트 폴더로 이동
 *
 * Usage: node scripts/render-video.js <project-folder>
 * Example: node scripts/render-video.js "test-ltx/output/2026-02-03_AI가-바꾸는-미래-직업"
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// 색상 출력 헬퍼
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
};

const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  warn: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
};

// 메인 함수
async function main() {
  const projectFolder = process.argv[2];

  if (!projectFolder) {
    log.error('프로젝트 폴더를 지정해주세요.');
    console.log('\nUsage: node scripts/render-video.js <project-folder>');
    console.log('Example: node scripts/render-video.js "test-ltx/output/2026-02-03_AI가-바꾸는-미래-직업"');
    process.exit(1);
  }

  const rootDir = process.cwd();
  const projectPath = path.resolve(rootDir, projectFolder);
  const publicOutputDir = path.join(rootDir, 'public', 'output');

  // 1. 프로젝트 폴더 확인
  log.info(`프로젝트 폴더 확인: ${projectPath}`);

  if (!fs.existsSync(projectPath)) {
    log.error(`프로젝트 폴더가 존재하지 않습니다: ${projectPath}`);
    process.exit(1);
  }

  // 2. public/output 폴더 생성 및 정리
  log.info('public/output 폴더 준비 중...');

  if (fs.existsSync(publicOutputDir)) {
    fs.rmSync(publicOutputDir, { recursive: true, force: true });
  }
  fs.mkdirSync(publicOutputDir, { recursive: true });

  // 3. 에셋 복사 (이미지, 오디오)
  log.info('에셋 복사 중...');

  const imagesDir = path.join(projectPath, 'images');
  const audioDir = path.join(projectPath, 'audio');

  // 이미지 복사
  if (fs.existsSync(imagesDir)) {
    const images = fs.readdirSync(imagesDir);
    images.forEach(file => {
      const src = path.join(imagesDir, file);
      const dest = path.join(publicOutputDir, file);
      fs.copyFileSync(src, dest);
      log.success(`복사됨: ${file}`);
    });
  } else {
    // 레거시 구조 지원 (images/ 폴더 없이 직접 파일)
    const files = fs.readdirSync(projectPath);
    files.filter(f => f.endsWith('.png') || f.endsWith('.jpg')).forEach(file => {
      const src = path.join(projectPath, file);
      const dest = path.join(publicOutputDir, file);
      fs.copyFileSync(src, dest);
      log.success(`복사됨: ${file}`);
    });
  }

  // 오디오 복사
  if (fs.existsSync(audioDir)) {
    const audioFiles = fs.readdirSync(audioDir);
    audioFiles.forEach(file => {
      const src = path.join(audioDir, file);
      const dest = path.join(publicOutputDir, file);
      fs.copyFileSync(src, dest);
      log.success(`복사됨: ${file}`);
    });
  } else {
    // 레거시 구조 지원
    const files = fs.readdirSync(projectPath);
    files.filter(f => f.endsWith('.mp3') || f.endsWith('.wav')).forEach(file => {
      const src = path.join(projectPath, file);
      const dest = path.join(publicOutputDir, file);
      fs.copyFileSync(src, dest);
      log.success(`복사됨: ${file}`);
    });
  }

  // 4. pipeline_output.json 복사 (메타데이터)
  const metadataDir = path.join(projectPath, 'metadata');
  const pipelineOutputPath = fs.existsSync(metadataDir)
    ? path.join(metadataDir, 'pipeline_output.json')
    : path.join(projectPath, 'pipeline_output.json');

  if (fs.existsSync(pipelineOutputPath)) {
    fs.copyFileSync(pipelineOutputPath, path.join(publicOutputDir, 'pipeline_output.json'));
    log.success('복사됨: pipeline_output.json');
  }

  // 5. Remotion 렌더링 실행
  log.info('Remotion 렌더링 시작...');

  const tempOutputPath = path.join(rootDir, 'temp_video_output.mp4');

  try {
    execSync(
      `npx remotion render src/remotion/index.ts KrackerVideo --output "${tempOutputPath}"`,
      {
        stdio: 'inherit',
        cwd: rootDir,
      }
    );
    log.success('Remotion 렌더링 완료!');
  } catch (error) {
    log.error('Remotion 렌더링 실패');
    cleanup(publicOutputDir);
    process.exit(1);
  }

  // 6. 결과물을 프로젝트 폴더로 이동
  const videoDir = path.join(projectPath, 'video');
  if (!fs.existsSync(videoDir)) {
    fs.mkdirSync(videoDir, { recursive: true });
  }

  const finalVideoPath = path.join(videoDir, 'final_video_remotion.mp4');

  if (fs.existsSync(tempOutputPath)) {
    fs.renameSync(tempOutputPath, finalVideoPath);
    log.success(`영상 저장됨: ${finalVideoPath}`);
  }

  // 7. public/output 정리
  cleanup(publicOutputDir);

  console.log('\n' + '='.repeat(50));
  log.success('렌더링 완료!');
  console.log(`📁 결과물: ${finalVideoPath}`);
  console.log('='.repeat(50));
}

function cleanup(dir) {
  log.info('임시 파일 정리 중...');
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
  log.success('정리 완료');
}

main().catch(err => {
  log.error(err.message);
  process.exit(1);
});
