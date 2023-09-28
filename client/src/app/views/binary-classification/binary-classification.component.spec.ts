import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BinaryClassificationComponent } from './binary-classification.component';

describe('BinaryClassificationComponent', () => {
  let component: BinaryClassificationComponent;
  let fixture: ComponentFixture<BinaryClassificationComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [BinaryClassificationComponent]
    });
    fixture = TestBed.createComponent(BinaryClassificationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
