import { Component, inject, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { TfService } from '../../services/tf.service';

@Component({
  selector: 'app-tf-model-train',
  templateUrl: './tf-model-train.component.html',
  styleUrls: ['./tf-model-train.component.scss']
})
export class TfModelTrainComponent implements OnInit {
  private tf = inject(TfService);

  private testingFeatures?: tf.Tensor;
  private testingLabels?: tf.Tensor;
  private normalisedFeature?: { min: tf.Tensor, max: tf.Tensor, tensor: tf.Tensor };
  private normalisedLabels?: { min: tf.Tensor, max: tf.Tensor, tensor: tf.Tensor };
  private storageKey = 'localstorage://test-kc-house-price-regression';
  private points: any[] = [];

  model?: tf.LayersModel;
  loss?: string;
  predictionValue?: number;
  isTrainingInProgress = false;
  isLoadingInProgress = false;

  ngOnInit() {
    this.tf.init();
  }

  toggleVisor(): void {
    this.tf.vis.visor().toggle();
  }

  predict(): void {
    if (Number.isNaN(this.predictionValue)) {
      alert("Please enter a valid number");
      return;
    }

    this.tf.core.tidy(() => {
      const inputTensor = this.tf.core.tensor1d([this.predictionValue!]);
      const normalizedInput = this.normalise(inputTensor, this.normalisedFeature?.min, this.normalisedFeature?.max);
      console.log('normalizedInput', normalizedInput.tensor.dataSync());
      console.log('this.normalisedFeature?.min', this.normalisedFeature?.min.dataSync()[0]);
      console.log('this.normalisedFeature?.max', this.normalisedFeature?.max.dataSync()[0]);

      const normalizedOutputTensor = this.model?.predict(normalizedInput.tensor) as tf.Tensor<tf.Rank>;
      const outputTensor = this.denormalise(normalizedOutputTensor, this.normalisedLabels!.min, this.normalisedLabels!.max);
      console.log('this.normalisedLabels?.min', this.normalisedLabels?.min.dataSync()[0]);
      console.log('this.normalisedLabels?.max', this.normalisedLabels?.max.dataSync()[0]);

      const outputValue = outputTensor.dataSync()[0];
      console.log(`The predicted house price is: ${outputValue}`);

      this.plotPredictionLine();
    });
  }

  async load(): Promise<void> {
    this.isLoadingInProgress = true;

    const models = await this.tf.core.io.listModels();
    const modelInfo = models[this.storageKey];

    if (modelInfo) {
      this.model = await tf.loadLayersModel(this.storageKey);

      await this.tf.vis.show.modelSummary({ name: 'Model Summary' }, this.model);
      const layer = this.model.getLayer('', 0);
      await this.tf.vis.show.layer({ name: 'Layer 1' }, layer);
    } else {
      alert('Could not load: no saved model found');
    }

    this.isLoadingInProgress = false;
  }

  async save(): Promise<void> {
    const saveResults = await this.model?.save(this.storageKey)
    console.log(`Trained and saved ${saveResults!.modelArtifactsInfo.dateSaved}`);
    alert('Trained model has been saved!');
  }

  test(): void {
    this.testModel();
  }

  async train(): Promise<void> {
    this.isTrainingInProgress = true;

    this.createModel();

    if (!this.model) {
      return;
    }

    this.model.summary();
    this.tf.vis.show.modelSummary({ name: `Model Summary`, tab: `Model` }, this.model);
    this.tf.vis.show.layer({ name: `Layer 1`, tab: `Model Inspection` }, this.model.getLayer('', 0));

    const houseSalesDataset = tf.data.csv('assets/kc_house_data.csv');

    const pointsDataset = houseSalesDataset.map((record: any) => ({ x: record.sqft_living, y: record.price }));
    const points = await pointsDataset.toArray();
    this.points = [...points];
    console.log('points.length', this.points.length);
    points.pop();
    tf.util.shuffle(points); // shuffling the array

    // const pointsBatch = pointsDataset.batch(100);
    // await pointsBatch.forEachAsync( e => console.log(e))

    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);
    featureTensor.print();

    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);
    labelTensor.print();

    // this.plot(points, 'Square Feet');

    this.normalisedFeature = this.normalise(featureTensor);
    this.normalisedLabels = this.normalise(labelTensor);
    this.normalisedFeature.tensor.print();
    this.normalisedLabels.tensor.print();

    const [trainingFeatures, testingFeatures] = tf.split(this.normalisedFeature.tensor, 2);
    const [trainingLabels, testingLabels] = tf.split(this.normalisedLabels.tensor, 2);

    this.testingFeatures = testingFeatures;
    this.testingLabels = testingLabels;

    await this.trainModel(this.model, trainingFeatures, trainingLabels);

    this.isTrainingInProgress = false;
  }

  private createModel(): void {
    let model: tf.LayersModel;

    if (!this.model) {
      model = this.tf.core.sequential();

      // Layer 1
      (model as tf.Sequential).add(this.tf.core.layers.dense({
        inputDim: 1,
        units: 1,
        activation: 'linear',
        useBias: true,
      }));

      this.model = model;
    }

    // Layer 2
    // model.add(this.tf.core.layers.dense({
    //   inputDim: 1,
    //   units: 3,
    //   activation: 'linear',
    //   useBias: false,
    // }));

    const optimizer = this.tf.core.train.sgd(0.1);
    this.model.compile({
      optimizer,
      loss: 'meanSquaredError'
    });
  }

  private trainModel(model: tf.LayersModel, trainingFeatureTensor: tf.Tensor, trainingLabelTensor: tf.Tensor) {
    const { onBatchEnd, onEpochEnd } = this.tf.vis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss']);

    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
      batchSize: 1000,
      epochs: 10,
      shuffle: true,
      validationSplit: 0.2,
      callbacks: {
        onBatchEnd,
        onEpochEnd,
        onEpochBegin: () => {
          this.tf.vis.show.layer({ name: `Layer 1` }, model.getLayer('', 0));
        }
      } as any,
    });
  }

  private normalise(tensor: tf.Tensor, prevMin?: tf.Tensor, prevMax?: tf.Tensor) {
    const min = prevMin ?? tensor.min();
    const max = prevMax ?? tensor.max();
    const normalisedTensor = tensor.sub(min).div(max.sub(min));
    return { tensor: normalisedTensor, min, max };
  }

  private denormalise(tensor: tf.Tensor, min: tf.Tensor, max: tf.Tensor) {
    return tensor.mul(max.sub(min)).add(min);
  }

  private testModel() {
    if (!this.model || !this.testingFeatures || !this.testingLabels) {
      alert('Model is not trained');
      return;
    }

    const lossTensor: tf.Scalar = this.model.evaluate(this.testingFeatures, this.testingLabels) as tf.Scalar;
    console.log(lossTensor);
    this.loss = Array.from(lossTensor.dataSync())[0].toPrecision(5);
    console.log(`Testing Loss: ${this.loss}`);
  }

  private plot(points: any, featureName: string, predictedPointsArray: any = null) {
    const values = [points];
    const series = ["original"];

    if (Array.isArray(predictedPointsArray)) {
      values.push(predictedPointsArray);
      series.push("predicted");
    }

    tfvis.render.scatterplot(
      { name: `${featureName} vs House Price` },
      { values: values, series: series },
      {
        xLabel: featureName,
        yLabel: 'Price',
      }
    );
  }

  async plotPredictionLine (): Promise<void> {
    const [xs, ys] = tf.tidy(() => {

      const normalisedXs = tf.linspace(0, 1, 100);
      const normalisedYs = this.model!.predict(normalisedXs.reshape([100, 1])) as tf.Tensor;

      const xs = this.denormalise(normalisedXs, this.normalisedFeature!.min, this.normalisedFeature!.max);
      const ys = this.denormalise(normalisedYs, this.normalisedLabels!.min, this.normalisedLabels!.max);

      return [ xs.dataSync(), ys.dataSync() ];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: ys[i]}
    });

    await this.plot(this.points, "Square feet", predictedPoints);
  }
}
