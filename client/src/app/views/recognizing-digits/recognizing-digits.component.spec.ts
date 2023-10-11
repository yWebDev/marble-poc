import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecognizingDigitsComponent } from './recognizing-digits.component';

describe('RecognizingDigitsComponent', () => {
  let component: RecognizingDigitsComponent;
  let fixture: ComponentFixture<RecognizingDigitsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [RecognizingDigitsComponent]
    });
    fixture = TestBed.createComponent(RecognizingDigitsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
