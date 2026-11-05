# 网页交互测试

在这个目录执行`npm install`和`npm test`。测试使用本机Microsoft Edge，以无界面方式运行；也可以通过`CATVISION_TEST_BROWSER`选择已安装的其他Chromium浏览器。Node.js需要支持内置的`node:test`。

测试打开实际的`web/index.html`，延迟接口返回后拖入另一张图片，检查文件名、结果和问答是否仍属于原任务。另有错误任务ID、错误图片任务的返回案例。接口使用固定数据，不加载模型，也不把这些结果计入识别准确率。后端任务元数据由`tests/test_image_question_api.py`单独检查。
