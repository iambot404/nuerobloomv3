/**
 * Automated Verification & Fact-Checking Suite: Dysgraphia Recognition Engine
 */

const engine = require('./static/dysgraphia_recognizer.js');

let passCount = 0;
let failCount = 0;

function assert(condition, message) {
    if (condition) {
        console.log(`  ✓ PASS: ${message}`);
        passCount++;
    } else {
        console.error(`  ✗ FAIL: ${message}`);
        failCount++;
    }
}

console.log('====================================================');
console.log('1. KINEMATICS & STRAIGHT LINE RECOGNITION');
console.log('====================================================');

// Perfectly horizontal straight line
const perfectLine = [{ points: [[20, 150, 0], [250, 150, 250], [480, 150, 500]] }];
const lineKin = engine.analyzeKinematics(perfectLine, 150);
assert(lineKin.straightness >= 0.95, `Straightness >= 0.95 (actual: ${lineKin.straightness})`);
assert(lineKin.tremor_stability >= 0.90, `Tremor stability >= 0.90 (actual: ${lineKin.tremor_stability})`);
assert(lineKin.rmse <= 1.0, `RMSE <= 1.0px (actual: ${lineKin.rmse})`);

const lineShape = engine.recognizeShape(perfectLine);
assert(lineShape.shape === 'Straight Line', `Shape classified as 'Straight Line' (actual: '${lineShape.shape}')`);
assert(lineShape.confidence >= 90, `Line confidence >= 90 (actual: ${lineShape.confidence})`);

console.log('\n====================================================');
console.log('2. GEOMETRIC SHAPE RECOGNITION (SQUARE, TRIANGLE, CIRCLE)');
console.log('====================================================');

// Square
const squareStrokes = [{ points: [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]] }];
const squareRes = engine.recognizeShape(squareStrokes, 'Square');
assert(squareRes.shape === 'Square', `Square recognized as 'Square' (actual: '${squareRes.shape}')`);
assert(squareRes.confidence >= 80, `Square confidence >= 80 (actual: ${squareRes.confidence})`);

// Triangle
const triStrokes = [{ points: [[150, 100], [220, 220], [80, 220], [150, 100]] }];
const triRes = engine.recognizeShape(triStrokes, 'Triangle');
assert(triRes.shape === 'Triangle', `Triangle recognized as 'Triangle' (actual: '${triRes.shape}')`);
assert(triRes.confidence >= 80, `Triangle confidence >= 80 (actual: ${triRes.confidence})`);

// Circle
const circlePts = [];
for (let a = 0; a <= 2 * Math.PI; a += 0.15) {
    circlePts.push([150 + 50 * Math.cos(a), 150 + 50 * Math.sin(a)]);
}
const circleRes = engine.recognizeShape([{ points: circlePts }], 'Circle');
assert(circleRes.shape === 'Circle', `Circle recognized as 'Circle' (actual: '${circleRes.shape}')`);
assert(circleRes.confidence >= 85, `Circle confidence >= 85 (actual: ${circleRes.confidence})`);

console.log('\n====================================================');
console.log('3. ALPHABET RECOGNITION (A, B, P, C, D, E)');
console.log('====================================================');

// Letter A
const letterA = [
    { points: [[150, 200], [175, 100]] },
    { points: [[175, 100], [200, 200]] },
    { points: [[162, 150], [188, 150]] }
];
const resA = engine.recognizeLetter(letterA, 'A');
assert(resA.isMatch === true, `Letter A isMatch == true (recognized: '${resA.letter}')`);
assert(resA.letter === 'A', `Letter A classified as 'A' (actual: '${resA.letter}')`);
assert(resA.accuracy >= 80, `Letter A accuracy >= 80 (actual: ${resA.accuracy})`);

// Letter B
const letterB = [
    { points: [[100, 100], [100, 200]] },
    { points: [[100, 100], [130, 125], [100, 150]] },
    { points: [[100, 150], [135, 175], [100, 200]] }
];
const resB = engine.recognizeLetter(letterB, 'B');
assert(resB.isMatch === true, `Letter B isMatch == true (recognized: '${resB.letter}')`);
assert(resB.letter === 'B', `Letter B classified as 'B' (actual: '${resB.letter}')`);
assert(resB.accuracy >= 75, `Letter B accuracy >= 75 (actual: ${resB.accuracy})`);

// Letter P
const letterP = [
    { points: [[100, 100], [100, 200]] },
    { points: [[100, 100], [130, 125], [100, 150]] }
];
const resP = engine.recognizeLetter(letterP, 'P');
assert(resP.isMatch === true, `Letter P isMatch == true (recognized: '${resP.letter}')`);
assert(resP.letter === 'P', `Letter P classified as 'P' (actual: '${resP.letter}')`);

// Letter C
const letterCPts = [];
for (let a = 0.5 * Math.PI; a <= 1.5 * Math.PI; a += 0.1) {
    letterCPts.push([150 + 40 * Math.cos(a), 150 + 40 * Math.sin(a)]);
}
const resC = engine.recognizeLetter([{ points: letterCPts }], 'C');
assert(resC.isMatch === true, `Letter C isMatch == true (recognized: '${resC.letter}')`);
assert(resC.letter === 'C', `Letter C classified as 'C' (actual: '${resC.letter}')`);

console.log('\n====================================================');
console.log(`TOTAL: ${passCount} PASSED, ${failCount} FAILED`);
console.log('====================================================');

if (failCount > 0) {
    process.exit(1);
} else {
    process.exit(0);
}
