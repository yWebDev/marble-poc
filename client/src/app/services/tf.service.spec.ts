import { TestBed } from '@angular/core/testing';

import { TfService } from './tf.service';

describe('TfService', () => {
  let service: TfService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TfService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
