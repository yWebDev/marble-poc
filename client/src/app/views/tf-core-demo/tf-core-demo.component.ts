import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-tf-core-demo',
  templateUrl: './tf-core-demo.component.html',
  styleUrls: ['./tf-core-demo.component.scss']
})
export class TfCoreDemoComponent implements OnInit {
  ngOnInit(): void {
    console.log('Task 1');
    this.calcSimpleOperation();
    console.log('Task 2');
    this.calcEquations();
    console.log('Task 3');
    this.normalizeTensor();
    console.log('Task 4');
    this.manageMemory();
  }

  private calcSimpleOperation(): void {
    const xs = tf.tensor1d([1, 2, 3]);
    const ys = xs.mul(tf.scalar(5));
    ys.print();

    let xs2d = xs.as2D(1, 3);
    xs2d.print();

    xs2d = xs2d.add(1);
    xs2d.print();

    xs2d = xs2d.sub(2);
    xs2d.print();

    xs2d = xs2d.add(ys);
    xs2d.print();

    xs2d = xs2d.add(xs2d);
    xs2d.print();
  }

  private calcEquations(): void {
    const getYs = (xs: tf.Tensor1D, m: number, c: number): tf.Tensor1D => xs.mul(m).add(c);

    const t1 = tf.tensor1d([1, 5, 10]);
    const t2 = getYs(t1, 2, 1);
    t2.print();
  }

  private normalizeTensor(): void {
    const t3 = tf.tensor1d([25, 76, 4, 23, -5, 22]);
    const max = t3.max(); // 76
    const min = t3.min(); // -5
    const t4 = t3.sub(min).div(max.sub(min));
    t4.print();
  }

  private manageMemory(): void {
    let tfc = [];
    for (let i = 0; i < 100; i++) {
      tfc.push(tf.tensor1d([1, 2, 3]));
      console.log(tf.memory().numTensors);
    }
    tf.dispose(tfc);

    const loop = () => {
      for (let i = 0; i < 100; i++) {
        tf.tensor1d([4, 5, 6]);
        console.log(tf.memory().numTensors);
      }
    };

    tf.tidy(loop);
    console.log(tf.memory().numTensors);
  }
}
