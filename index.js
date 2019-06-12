const {Network, DigitRecognition} = require("digits-recognition-neural-network");

const PARAM_NAMES = {
	HS: "hidden-size",
	EPOCHS: "epochs",
	LR: "learning-rate",
	PATH: "path",
	TRAIN: "train-size",
	MOMENT: "moment"
};

init();

async function init() {
	const hs = getParam(PARAM_NAMES.HS, true) || 100;
	const epochs = getParam(PARAM_NAMES.EPOCHS, true) || 50;
	const learningRate = getParam(PARAM_NAMES.LR, true) || 0.3;
	const path = getParam(PARAM_NAMES.PATH);
	const trainSize = getParam(PARAM_NAMES.TRAIN, true) || 1000;
	const moment = getParam(PARAM_NAMES.MOMENT, true) || 0;

	await createModel(hs, epochs, learningRate, path, trainSize, moment);
}

async function createModel(hiddenSize, epochs, learningRate, modelPath, trainSize, moment) {
	const recognizer = new DigitRecognition(hiddenSize, learningRate, moment);
	recognizer.train(trainSize, epochs);

	await Network.serialize(recognizer.network, modelPath);
}

function getParam(param, isNumber = false) {
	const pattern = new RegExp(`-{0,2}${param}=`);
	const parsedParam = process.argv.find((item) => pattern.test(item));
	const splitted = parsedParam ? parsedParam.split(pattern) : null;
	const value = splitted && splitted.length
		? splitted[splitted.length - 1]
		: null;

	if (isNumber) {
		return value ? parseInt(value, 10) : null;
	}

	return value;
}
