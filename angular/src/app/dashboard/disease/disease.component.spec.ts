import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DiseaseComponent } from './disease.component';

describe('DiseaseComponent', () => {
  let component: DiseaseComponent;
  let fixture: ComponentFixture<DiseaseComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DiseaseComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DiseaseComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
