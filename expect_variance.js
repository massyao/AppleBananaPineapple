
const allNum = 10000000

const sampleNum = 10

const sampleTimes = 10

const allList = Array.from(new Array(allNum), (e, i) => Math.random())

const expect = allList.reduce((acc, cur) => acc + cur, 0) / allNum

const variance = allList.reduce((acc, cur) => acc + (expect - cur)*(expect - cur), 0) / allNum




const oneExperiment = () => {
  const sampleList = Array.from(new Array(sampleNum), (e, i) => allList[Math.floor(Math.random() * allNum)])
  const sampleExpect = sampleList.reduce((acc, cur) => acc + cur, 0) / sampleNum

  const sampleVariance = sampleList.reduce((acc, cur) => acc + (sampleExpect - cur)*(sampleExpect - cur), 0) / sampleNum
  
  const sampleCorrectVariance = sampleList.reduce((acc, cur) => acc + (sampleExpect - cur)*(sampleExpect - cur), 0) / (sampleNum - 1)
  return {
    sampleExpect,
    sampleVariance,
    sampleCorrectVariance,
  }
}

const {
  sampleExpect,
  sampleVariance,
  sampleCorrectVariance,
} = (() => {
  const experimentList = Array.from(new Array(sampleTimes), (e, i) => oneExperiment())
  return {
    sampleExpect: experimentList.reduce((acc, cur) => acc + cur.sampleExpect, 0) / sampleTimes,
    sampleVariance: experimentList.reduce((acc, cur) => acc + cur.sampleVariance, 0) / sampleTimes,
    sampleCorrectVariance: experimentList.reduce((acc, cur) => acc + cur.sampleCorrectVariance, 0) / sampleTimes,
  }
})() 


console.table({
  expect,
  variance,
  sampleExpect,
  sampleVariance,
  sampleCorrectVariance,
  expectDiff: expect - sampleExpect,
  expectDiffRate: (expect - sampleExpect) / expect,
  varianceDiff: variance - sampleVariance,
  varianceDiffRate: (variance - sampleVariance) / variance,
  correctVarianceDiff: variance - sampleCorrectVariance,
  correctVarianceDiffRate: (variance - sampleCorrectVariance) / variance,
})
