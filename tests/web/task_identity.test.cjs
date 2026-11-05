const {test, before, after} = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

const imageId = 'a'.repeat(32);
const questionId = 'c'.repeat(32);
const imageHash = '1'.repeat(64);
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1sAAAAASUVORK5CYII=', 'base64');
const photo = name => ({name, mimeType: 'image/png', buffer: png});
const deferred = () => {let resolve; const promise = new Promise(r => {resolve = r}); return {promise, resolve}};
let browser, server, origin;

before(async () => {
  server = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.end(fs.readFileSync(path.join(__dirname, '../../web/index.html')));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({channel: process.env.CATVISION_TEST_BROWSER || 'msedge', headless: true});
});
after(async () => {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
});

function completedImage() {
  return {id: imageId, kind: 'recognition', filename: 'A.png', input_sha256: imageHash,
    state: 'completed', input_url: `/files/${imageId}/input.png`,
    result: {task_id: imageId, input_sha256: imageHash, total_seconds: 1,
      traditional: {status: 'ok', breed_zh: '布偶猫', breed_score: .91, inference_seconds: .1},
      vlm: {status: 'ok', breed_zh: '布偶猫', features: ['白色长毛'], inference_seconds: 1}}};
}

async function openPage(t, {holdImage = false, holdAnswer = false, changeImage = x => x, changeAnswer = x => x} = {}) {
  const page = await browser.newPage();
  page.setDefaultTimeout(4000);
  const imageGate = deferred(), answerGate = deferred(), submitted = deferred(), asked = deferred();
  if (!holdImage) imageGate.resolve();
  if (!holdAnswer) answerGate.resolve();
  t.after(async () => {imageGate.resolve(); answerGate.resolve(); await page.close()});
  await page.route('**/api/**', async route => {
    const request = route.request(), url = new URL(request.url());
    let data;
    if (url.pathname === '/api/info') data = {breeds: ['布偶猫'], models_preloaded: true};
    else if (url.pathname === '/api/recognize') {
      submitted.resolve(request.postDataBuffer().toString().includes('A.png'));
      data = {id: imageId, kind: 'recognition', state: 'queued', filename: 'A.png', input_sha256: imageHash};
    } else if (url.pathname === `/api/jobs/${imageId}/questions`) {
      asked.resolve(request.postDataJSON());
      data = {id: questionId, kind: 'image_question', state: 'queued', parent_id: imageId, input_sha256: imageHash};
    } else if (url.pathname === `/api/jobs/${imageId}`) {
      await imageGate.promise;
      data = changeImage(completedImage());
    } else if (url.pathname === `/api/jobs/${questionId}`) {
      await answerGate.promise;
      data = changeAnswer({id: questionId, parent_id: imageId, input_sha256: imageHash, kind: 'image_question',
        state: 'completed', question: '耳朵是什么样？',
        result: {task_id: questionId, parent_id: imageId, input_sha256: imageHash, answer: '耳朵直立。', inference_seconds: .2}});
    } else throw Error(`Unexpected request: ${request.method()} ${url.pathname}`);
    await route.fulfill({json: data});
  });
  await page.goto(origin);
  await page.waitForFunction(() => document.getElementById('status-text').textContent.includes('模型已准备好'));
  await page.setInputFiles('#file', photo('A.png'));
  await page.click('#run');
  assert.equal(await submitted.promise, true, 'The uploaded file must be A');
  return {page, imageGate, answerGate, asked};
}

async function dropB(page) {
  await page.locator('#drop').evaluate(element => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(['different-image'], 'B.png', {type: 'image/png'}));
    element.dispatchEvent(new DragEvent('drop', {dataTransfer: transfer, bubbles: true, cancelable: true}));
  });
}

test('recognition locks file selection and drops until A is displayed', async t => {
  const {page, imageGate} = await openPage(t, {holdImage: true});
  await dropB(page);
  assert.equal(await page.locator('#file-name').innerText(), 'A.png');
  assert.equal(await page.locator('#file').isDisabled(), true);
  assert.equal(await page.locator('#choose').isDisabled(), true);
  imageGate.resolve();
  await page.waitForFunction(() => document.getElementById('question-area').hidden === false);
  assert.equal(await page.locator('#file-name').innerText(), 'A.png');
  assert.match(await page.locator('#traditional-body').innerText(), /布偶猫/);
  await page.setInputFiles('#file', photo('B.png'));
  assert.equal(await page.locator('#file-name').innerText(), 'B.png');
  assert.equal(await page.locator('#question-area').isHidden(), true);
  assert.doesNotMatch(await page.locator('#traditional-body').innerText(), /布偶猫/);
  assert.doesNotMatch(await page.locator('#traditional-score').innerText(), /91/);
});

test('image questions lock drops and use the completed image task', async t => {
  const {page, answerGate, asked} = await openPage(t, {holdAnswer: true});
  await page.waitForFunction(() => document.getElementById('question-area').hidden === false);
  await page.fill('#question-input', '耳朵是什么样？');
  await page.click('#ask');
  assert.deepEqual(await asked.promise, {question: '耳朵是什么样？'});
  await dropB(page);
  assert.equal(await page.locator('#file-name').innerText(), 'A.png');
  assert.equal(await page.locator('#file').isDisabled(), true);
  answerGate.resolve();
  await page.waitForFunction(() => document.getElementById('answers').textContent.includes('耳朵直立'));
  assert.equal(await page.locator('#file-name').innerText(), 'A.png');
  assert.equal(await page.locator('#choose').isDisabled(), false);
});

test('recognition rejects a result carrying another task ID', async t => {
  const {page} = await openPage(t, {changeImage: job => ({...job, id: 'b'.repeat(32)})});
  await page.waitForFunction(() => !document.getElementById('choose').disabled);
  assert.equal(await page.locator('#question-area').isHidden(), true);
  assert.doesNotMatch(await page.locator('#traditional-body').innerText(), /布偶猫/);
  assert.match(await page.locator('#status-text').innerText(), /任务.*不匹配/);
});

test('recognition rejects mismatched result identity even if envelope ID matches', async t => {
  const {page} = await openPage(t, {changeImage: job => ({...job, result: {...job.result, task_id: 'b'.repeat(32)}})});
  await page.waitForFunction(() => !document.getElementById('choose').disabled);
  assert.equal(await page.locator('#question-area').isHidden(), true);
  assert.doesNotMatch(await page.locator('#traditional-body').innerText(), /布偶猫/);
});

test('a question answer for another parent image is never displayed', async t => {
  const {page} = await openPage(t, {changeAnswer: job => ({...job, parent_id: 'b'.repeat(32)})});
  await page.waitForFunction(() => document.getElementById('question-area').hidden === false);
  await page.fill('#question-input', '耳朵是什么样？');
  await page.click('#ask');
  await page.waitForFunction(() => !document.getElementById('ask').disabled);
  assert.equal(await page.locator('#answers').innerText(), '');
  assert.match(await page.locator('#status-text').innerText(), /任务.*不匹配/);
});
