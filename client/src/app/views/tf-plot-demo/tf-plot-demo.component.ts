import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

@Component({
  selector: 'app-tf-plot-demo',
  templateUrl: './tf-plot-demo.component.html',
  styleUrls: ['./tf-plot-demo.component.scss']
})
export class TfPlotDemoComponent implements OnInit {
  ngOnInit() {
    tf.tidy(() => {
      this.run().then(() => console.log('Memory', tf.memory()));
    });
  }

  private async run(): Promise<void> {
    await tf.ready();
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
  }

  private plot(points: any, featureName: string) {
    tfvis.render.scatterplot(
      { name: `${featureName} vs House Price` },
      { values: [points], series: ['original'] },
      {
        xLabel: featureName,
        yLabel: 'Price',
      }
    );
  }

  private normalise(tensor: tf.Tensor) {
    const max = tensor.max();
    const min = tensor.min();
    const normalisedTensor = tensor.sub(min).div(max.sub(min));
    return { tensor: normalisedTensor, min, max };
  }

  private denormalise(tensor: tf.Tensor, min: tf.Tensor, max: tf.Tensor) {
    return tensor.mul(max.sub(min)).add(min);
  }
}
