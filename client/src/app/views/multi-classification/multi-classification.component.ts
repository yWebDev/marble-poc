import { Component, inject, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { TfService } from '../../services/tf.service';

@Component({
  selector: 'app-multi-classification',
  templateUrl: './multi-classification.component.html',
  styleUrls: ['./multi-classification.component.scss']
})
export class MultiClassificationComponent implements OnInit {
  private tf = inject(TfService);

  private testingFeatures?: tf.Tensor;
  private testingLabels?: tf.Tensor;
  private normalisedFeature?: { min: tf.Tensor, max: tf.Tensor, tensor: tf.Tensor };
  private normalisedLabels?: { min: tf.Tensor, max: tf.Tensor, tensor: tf.Tensor };
  private storageKey = 'localstorage://test-kc-house-multi';
  private points: any[] = [];

  model?: tf.LayersModel;
  loss?: string;
  predictionValue?: number;
  squareFeet?: number;
  price?: number;
  predictionOutput?: string;
  isTrainingInProgress = false;
  isLoadingInProgress = false;

  ngOnInit() {
    this.tf.init().then(() => {
      // this.run();
    });
  }

  toggleVisor(): void {
    this.tf.vis.visor().toggle();
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

    await this.plotPredictionHeatmap();

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

    const pointsDataset = houseSalesDataset

      .map((record: any) => ({
        x: record.sqft_living,
        y: record.price,
        class: record.bedrooms > 2 ? "3+" : record.bedrooms,
      }))
      .filter((record: any) => !!record.class);
    const points = await pointsDataset.toArray();

    // points.pop();
    tf.util.shuffle(points); // shuffling the array

    // const pointsBatch = pointsDataset.batch(100);
    // await pointsBatch.forEachAsync( e => console.log(e))

    const featureValues = points.map(p => [p.x, p.y]);
    const featureTensor = tf.tensor2d(featureValues);
    featureTensor.print();

    const labelValues = points.map(p => this.getClassIndex(p.class));
    const labelTensor = tf.tidy(() => tf.oneHot(tf.tensor1d(labelValues,  'int32'), 3));
    labelTensor.print();

    this.plotClasses(points, "Bedrooms");

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
    await this.plotPredictionHeatmap();

    this.isTrainingInProgress = false;
  }

  private createModel(): void {
    let model: tf.LayersModel;

    if (!this.model) {
      model = this.tf.core.sequential();

      // Layer 1
      (model as tf.Sequential).add(this.tf.core.layers.dense({
        units: 10,
        useBias: true,
        activation: 'softmax',
        inputDim: 2,
      }));

      // Layer 2
      (model as tf.Sequential).add(this.tf.core.layers.dense({
        units: 10,
        activation: 'softmax',
        useBias: true,
      }));

      // Layer 3
      (model as tf.Sequential).add(this.tf.core.layers.dense({
        units: 3,
        activation: 'softmax',
        useBias: true,
      }));

      this.model = model;
    }

    const optimizer = this.tf.core.train.sgd(0.1);
    this.model.compile({
      optimizer,
      loss: 'categoricalCrossentropy'
    });
  }

  private trainModel(model: tf.LayersModel, trainingFeatureTensor: tf.Tensor, trainingLabelTensor: tf.Tensor) {
    const { onBatchEnd, onEpochEnd } = this.tf.vis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss']);

    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
      batchSize: 32,
      epochs: 10,
      shuffle: true,
      validationSplit: 0.2,
      callbacks: {
        onBatchEnd,
        onEpochEnd,
        onEpochBegin: () => {
          this.tf.vis.show.layer({ name: `Layer 1` }, model.getLayer('', 0));
          this.plotPredictionHeatmap();
        }
      } as any,
    });
  }

  private normalise(tensor: tf.Tensor, previousMin: tf.Tensor | tf.Tensor[] | null = null, previousMax: tf.Tensor | tf.Tensor[] | null = null) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
    if (featureDimensions && featureDimensions > 1) {
      // More than one feature
      // Split into separate tensors
      const features = tf.split(tensor, featureDimensions, 1);

      // Normalise and find min/max values for each feature
      const normalisedFeatures = features.map((featureTensor, i) =>
        this.normalise(featureTensor, previousMin ? (previousMin as tf.Tensor[])[i] : null, previousMax ? (previousMax as tf.Tensor[])[i] : null));

      // Prepare return values
      // In this case the min and max properties will be arrays, with one
      // value for each feature
      const returnTensor: any = tf.concat(normalisedFeatures.map((f: any) => f.tensor), 1);
      const min: any = normalisedFeatures.map((f: any) => f.min);
      const max: any = normalisedFeatures.map((f: any) => f.max);
      return { tensor: returnTensor, min, max };
    } else {
      // Just one feature
      const min = previousMin || tensor.min();
      const max = previousMax || tensor.max();
      const normalisedTensor = tensor.sub(min as tf.Tensor).div((max as tf.Tensor).sub(min as tf.Tensor));
      return {
        tensor: normalisedTensor,
        min,
        max
      };
    }
  }

  private denormalise(tensor: tf.Tensor, min: tf.Tensor | tf.Tensor[], max: tf.Tensor | tf.Tensor[]) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
    if (featureDimensions && featureDimensions > 1) {
      // More than one feature
      // Split into separate tensors
      const features = tf.split(tensor, featureDimensions, 1);

      // Denormalise
      const denormalised: any = features.map((featureTensor, i) => this.denormalise(featureTensor, (min as tf.Tensor[])[i], (max as tf.Tensor[])[i]));

      const returnTensor = tf.concat(denormalised, 1);
      return returnTensor;
    } else {
      const denormalisedTensor = tensor.mul((max as tf.Tensor).sub((min as tf.Tensor))).add((min as tf.Tensor));
      return denormalisedTensor;
    }
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

  private async plotClasses(pointsArray: any[], classKey: string, size = 400, equalizeClassSizes = false): Promise<void> {
    // Add each class as a series
    const allSeries: any = {};
    pointsArray.forEach(p => {
      // Add each point to the series for the class it is in
      const seriesName = `${classKey}: ${p.class}`;
      let series = allSeries[seriesName];
      if (!series) {
        series = [];
        allSeries[seriesName] = series;
      }
      series.push(p);
    });

    if (equalizeClassSizes) {
      // Find smallest class
      let maxLength: any = null;
      Object.values(allSeries).forEach((series: any) => {
        if (maxLength === null || series.length < maxLength && series.length >= 100) {
          maxLength = series.length;
        }
      });
      // Limit each class to number of elements of smallest class
      Object.keys(allSeries).forEach(keyName => {
        allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
        if (allSeries[keyName].length < 100) {
          delete allSeries[keyName];
        }
      });
    }

    tfvis.render.scatterplot(
      {
        name: `Square feet vs House Price`,
        styles: { width: "100%" }
      },
      {
        values: Object.values(allSeries) as any,
        series: Object.keys(allSeries),
      },
      {
        xLabel: "Square feet",
        yLabel: "Price",
        height: size,
        width: size * 1.5,
      }
    );
  }

  private async run(): Promise<void> {
    await tf.ready();
    const houseSalesDataset = tf.data.csv('assets/kc_house_data.csv');

    const pointsDataset = houseSalesDataset

      .map((record: any) => ({
        x: record.sqft_living,
        y: record.price,
        class: record.bedrooms > 2 ? "3+" : record.bedrooms,
      }))
      .filter((record: any) => !!record.class);
    const points = await pointsDataset.toArray();

    // points.pop();
    tf.util.shuffle(points); // shuffling the array

    // const pointsBatch = pointsDataset.batch(100);
    // await pointsBatch.forEachAsync( e => console.log(e))

    const featureValues = points.map(p => [p.x, p.y]);
    const featureTensor = tf.tensor2d(featureValues);
    featureTensor.print();

    const labelValues = points.map(p => this.getClassIndex(p.class));
    const labelTensor = tf.tidy(() => tf.oneHot(tf.tensor1d(labelValues,  'int32'), 3));
    labelTensor.print();

    this.plotClasses(points, "Bedrooms");

    this.normalisedFeature = this.normalise(featureTensor);
    this.normalisedLabels = this.normalise(labelTensor);
    this.normalisedFeature.tensor.print();
    this.normalisedLabels.tensor.print();

    const [trainingFeatures, testingFeatures] = tf.split(this.normalisedFeature.tensor, 2);
    const [trainingLabels, testingLabels] = tf.split(this.normalisedLabels.tensor, 2);
  }

  private async plotPredictionHeatmap(name = "Predicted class", size = 400) {
    const [valuesPromise, xTicksPromise, yTicksPromise] = (() => {
      const gridSize = 50;
      const predictionColumns = [];
      // Heatmap order is confusing: columns first (top to bottom) then rows (left to right)
      // We want to convert that to a standard cartesian plot so invert the y values
      for (let colIndex = 0; colIndex < gridSize; colIndex++) {
        // Loop for each column, starting from the left
        const colInputs = [];
        const x = colIndex / gridSize;
        for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
          // Loop for each row, starting from the top
          const y = (gridSize - rowIndex) / gridSize;
          colInputs.push([x, y]);
        }

        const colPredictions = this.model!.predict(tf.tensor2d(colInputs));
        predictionColumns.push(colPredictions);
      }
      const valuesTensor = tf.stack(predictionColumns as tf.Tensor[]);

      const normalisedLabelsTensor = tf.linspace(0, 1, gridSize);
      const xTicksTensor = this.denormalise(normalisedLabelsTensor,
        (this.normalisedFeature!.min as unknown as tf.Tensor[])[0], (this.normalisedFeature!.max as unknown as tf.Tensor[])[0]);
      const yTicksTensor = this.denormalise(normalisedLabelsTensor.reverse(),
        (this.normalisedFeature!.min as unknown as tf.Tensor[])[1], (this.normalisedFeature!.max as unknown as tf.Tensor[])[1]);

      return [valuesTensor.array(), xTicksTensor.array(), yTicksTensor.array()] as any;
    })();

    const values = await valuesPromise;
    const xTicks = await xTicksPromise;
    const xTickLabels = xTicks.map((l: number) => (l / 1000).toFixed(1) + "k sqft");
    const yTicks = await yTicksPromise;
    const yTickLabels = yTicks.map((l: number) => "$" + (l / 1000).toFixed(0) + "k");
    const data = {
      values,
      xTickLabels,
      yTickLabels,
    };

    tf.unstack(values, 2).forEach((values, i) => {
      const data = {
        values,
        xTickLabels,
        yTickLabels,
      };

      tfvis.render.heatmap({
        name: `Bedrooms: ${this.getClassName(i)} (local)`,
        tab: "Predictions"
      }, data as any, {height: size});
      tfvis.render.heatmap({
        name: `Bedrooms: ${this.getClassName(i)} (full domain)`,
        tab: "Predictions"
      }, data as any, {height: size, domain: [0, 1]});
    });
  }

  predict(): void {
    if (!this.price || !this.squareFeet) {
      return;
    }
    tf.tidy(() => {
      const inputTensor = tf.tensor2d([[this.squareFeet!, this.price!]]);
      const normalisedInput = this.normalise(inputTensor, this.normalisedFeature!.min, this.normalisedFeature!.max);
      const normalisedOutputTensor = this.model!.predict(normalisedInput.tensor);
      const outputTensor = this.denormalise(normalisedOutputTensor as tf.Tensor, this.normalisedLabels!.min, this.normalisedLabels!.max);
      const outputs = outputTensor.dataSync();
      let outputString = "";
      for (let i = 0; i < 3; i++) {
        outputString += `Likelihood of having ${this.getClassName(i)} bedrooms is: ${(100*outputs[i]).toFixed(1)}%<br>`;
      }
      this.predictionOutput = outputString;
    });
  }

  private getClassIndex(className: string | number): 0 | 1 | 2 {
    if (className === 1 || className === "1") {
      return 0; // 1 bedroom
    }
    else if (className === 2 || className === "2") {
      return 1; // 2 bedrooms
    }
    else {
      return 2; // 3 bedrooms
    }
  }

  private getClassName(classIndex: number): string {
    if (classIndex === 2)
      return "3+";
    else
      return classIndex+1+'';
  }
}
