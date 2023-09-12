import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

@Injectable({
  providedIn: 'root'
})
export class TfService {
  get core(): typeof tf {
    return tf;
  }

  get vis(): typeof tfvis {
    return tfvis;
  }

  async init(): Promise<typeof tf> {
    await tf.ready();

    console.log('TF Version', tf.version);
    console.log('TF Backend', tf.getBackend());

    return tf;
  }

  tidy(func: (...args: unknown[]) => any): void {
    tf.tidy(func);
  }
}
