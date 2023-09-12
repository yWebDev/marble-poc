import { Component, inject, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { TfService } from '../../services/tf.service';

@Component({
  selector: 'app-tf-model-train',
  templateUrl: './tf-model-train.component.html',
  styleUrls: ['./tf-model-train.component.scss']
})
export class TfModelTrainComponent implements OnInit {
  private tf = inject(TfService);

  ngOnInit() {
    this.tf.init().then(() => this.run());
  }

  private async run(): Promise<void> {
    const model = this.createModel();
    model.summary();
    this.tf.vis.show.modelSummary({ name: `Model Summary`, tab: `Model` }, model);
    this.tf.vis.show.layer({ name: `Layer 1`, tab: `Model Inspection` }, model.getLayer('', 0));

    const houseSalesDataset = tf.data.csv('assets/kc_house_data.csv');

    const pointsDataset = houseSalesDataset.map((record: any) => ({ x: record.sqft_living, y: record.price }));
    const points = await pointsDataset.toArray();
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

    const normalisedFeature = this.normalise(featureTensor);
    const normalisedLabel = this.normalise(labelTensor);
    normalisedFeature.tensor.print();
    normalisedLabel.tensor.print();

    const [trainingFeatures, testingFeatures] = tf.split(normalisedFeature.tensor, 2);
    const [trainingLabels, testingLabels] = tf.split(normalisedLabel.tensor, 2);

    await this.trainModel(model, trainingFeatures, trainingLabels);
    this.testModel(model, trainingFeatures, trainingLabels);
  }

  private createModel() {
    const model = this.tf.core.sequential();

    // Layer 1
    model.add(this.tf.core.layers.dense({
      inputDim: 1,
      units: 1,
      activation: 'linear',
      useBias: true,
    }));

    // Layer 2
    // model.add(this.tf.core.layers.dense({
    //   inputDim: 1,
    //   units: 3,
    //   activation: 'linear',
    //   useBias: false,
    // }));

    const optimizer = this.tf.core.train.sgd(0.1);
    model.compile({
      optimizer,
      loss: 'meanSquaredError'
    });

    return model;
  }

  private trainModel(model: tf.LayersModel, trainingFeatureTensor: tf.Tensor, trainingLabelTensor: tf.Tensor) {
    const { onBatchEnd, onEpochEnd } = this.tf.vis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss']);

    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
      batchSize: 8000,
      epochs: 20,
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

  private normalise(tensor: tf.Tensor) {
    const max = tensor.max();
    const min = tensor.min();
    const normalisedTensor = tensor.sub(min).div(max.sub(min));
    return { tensor: normalisedTensor, min, max };
  }

  private testModel(model: tf.LayersModel, testingFeatureTensor: tf.Tensor, testingLabelTensor: tf.Tensor) {
    const lossTensor: tf.Scalar = model.evaluate(testingFeatureTensor, testingLabelTensor) as tf.Scalar;
    console.log(lossTensor);
    const loss = lossTensor.dataSync();
    console.log(`Testing Loss: ${loss}`);
  }
}
